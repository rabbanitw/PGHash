import numpy as np
import tensorflow as tf
import time
from misc import compute_accuracy_lsh
import resource
import os
import datetime


def get_memory(filename):
    mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    with open(filename, 'a') as f:
        # Dump timestamp, PID and amount of RAM.
        f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

def get_partial_label(sparse_y, sub_idx, batch_size, full_num_labels):
    '''
    Takes a sparse full label and converts it into a dense sub-label corresponding to the output nodes in the
    sub-architecture for a given device
    :param sparse_y: Sparse full labels
    :param sub_idx: Indices for which output nodes are used/activated in the sub-architecture
    :param batch_size: Batch size
    :param full_num_labels: Total number of output nodes
    :return: Dense sub-label corresponding to given output nodes
    '''
    true_idx = sparse_y.indices.numpy()
    y_true = np.zeros((batch_size, full_num_labels))
    for i in true_idx:
        y_true[i[0], i[1]] = 1
    y_true = y_true[:, sub_idx]
    nz = np.count_nonzero(y_true, axis=1).reshape(batch_size, 1)
    nz[nz == 0] = 1
    y_true = y_true / nz
    return tf.convert_to_tensor(y_true, dtype=tf.float32)


def slide_partial_label(sparse_y, sub_idx, batch_size, full_num_labels):
    '''
    Takes a sparse full label and converts it into a dense sub-label corresponding to the output nodes in the
    sub-architecture for a given device
    :param sparse_y: Sparse full labels
    :param sub_idx: Indices for which output nodes are used/activated in the sub-architecture
    :param batch_size: Batch size
    :param full_num_labels: Total number of output nodes
    :return: Dense sub-label corresponding to given output nodes
    '''

    mask = np.zeros((batch_size, full_num_labels))
    for j in range(batch_size):
        mask[j, sub_idx[j]] = 1
    return tf.sparse.to_dense(sparse_y), tf.convert_to_tensor(mask, dtype=tf.float32)


def pg_train(rank, size, Method, train_data, test_data, losses, top1, test_top1, recorder, args, num_labels,
             num_features):

    # parameters
    total_batches = 0
    iterations = 1
    test_acc = np.NaN
    comm_time = 0
    if args.cr == 1:
        smartavg = False
    else:
        smartavg = True

    for epoch in range(args.epochs):
        print("\nStart of epoch %d" % (epoch,))

        # shuffle training data each epoch
        train_data.shuffle(len(train_data))

        # iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in train_data:

            batches_per_q = np.ceil(x_batch_train.shape[0] / args.train_bs).astype(np.int32)

            lsh_init = time.time()
            # update full model
            Method.update_full_model(Method.model)
            # compute LSH
            cur_idx = Method.lsh_initial(Method.model, x_batch_train)
            # send indices to root (server)
            Method.exchange_idx()
            # update model
            Method.update_model()
            # when updating model I need to restart optimizer for some reason...
            optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)  # might need to restart optimizer
            # reset the correct iteration after re-initializing
            optimizer.iterations = tf.Variable(iterations-1, dtype=tf.int64, name='iter')

            lsh_time = time.time() - lsh_init

            for sub_batch in range(batches_per_q):

                # compute test accuracy every X steps
                if iterations % args.steps_per_test == 0:
                    if rank == 0:
                        Method.update_full_model(Method.model)
                        test_acc = Method.test_full_model(test_data, test_top1)
                        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                        recorder.add_testacc(test_acc)
                        test_top1.reset()

                x = tf.sparse.slice(x_batch_train, start=[sub_batch * args.train_bs, 0],
                                    size=[args.train_bs, num_features])
                y = tf.sparse.slice(y_batch_train, start=[sub_batch * args.train_bs, 0],
                                    size=[args.train_bs, num_labels])

                init_time = time.time()

                # communicate models amongst devices (if multiple devices are present)
                if size > 1:
                    model, comm_time = Method.communicate(Method.model, smart=smartavg)

                # transform sparse label to dense sub-label
                batch = x.get_shape()[0]
                y_true = get_partial_label(y, cur_idx, batch, num_labels)

                # perform gradient update
                with tf.GradientTape() as tape:
                    y_pred = Method.model(x)
                    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
                    # issue computing the loss when y_true is 1 but mask made it 0 (since it didn't predict it)
                    # i need to make that value 1
                    # multiply y_true by mask...
                grads = tape.gradient(loss_value, Method.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, Method.model.trainable_weights))

                # compute accuracy (top 1) and loss for the minibatch
                rec_init = time.time()
                losses.update(np.array(loss_value), batch)
                acc1 = compute_accuracy_lsh(y_pred, y, cur_idx, num_labels)
                top1.update(acc1, batch)
                record_time = time.time() - rec_init
                comp_time = (time.time() - init_time) - (record_time + comm_time)

                if sub_batch != 0:
                    lsh_time = 0

                # store and save accuracy and loss values
                recorder.add_new(comp_time + comm_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                                 loss_value.numpy(), top1.avg, losses.avg)
                recorder.save_to_file()

                # log every X batches
                total_batches += batch
                if iterations % 5 == 0:
                    print(
                        "(Rank %d) Step %d: Epoch Time %f, Comm Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, "
                        "[%d Total Samples]" % (rank, iterations, (comp_time + comm_time), comm_time,
                                                loss_value.numpy(), acc1, total_batches)
                    )

                iterations += 1

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()


def slide_train(rank, Method, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args, num_labels):

    # parameters
    total_batches = 0
    test_acc = np.NaN
    acc1_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    full_idx = np.arange(num_labels)
    iterations = 1

    # update indices with new current index
    Method.ci = np.arange(num_labels)
    Method.bias_idx = Method.ci + Method.bias_start
    # get model
    model = Method.model

    for epoch in range(args.epochs):
        print("\nStart of epoch %d" % (epoch,))

        # shuffle training data each epoch
        train_data.shuffle(len(train_data))

        # iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in train_data:

            # compute test accuracy every X steps
            if iterations % args.steps_per_test == 0:
                if rank == 0:
                    test_acc = Method.test_full_model(test_data, test_top1)
                    print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                    recorder.add_testacc(test_acc)
                    test_top1.reset()

            if iterations % args.steps_per_lsh == 0 or iterations == 1:
                lsh_init = time.time()
                # compute LSH hash tables
                Method.lsh_get_hash()
                # compute best neurons for this batch of data via SLIDE
                batch_idxs = Method.lsh(x_batch_train)
                concat_idx = np.concatenate(batch_idxs)
                non_active_idx = np.setdiff1d(full_idx, concat_idx)
                lsh_time = time.time() - lsh_init
            else:
                lsh_init = time.time()
                # compute best neurons for all samples in the batch of data via SLIDE
                batch_idxs = Method.lsh(x_batch_train)
                concat_idx = np.concatenate(batch_idxs)
                non_active_idx = np.setdiff1d(full_idx, concat_idx)
                lsh_time = time.time() - lsh_init

            init_time = time.time()

            # transform sparse label to dense sub-label
            batch = x_batch_train.get_shape()[0]
            y_true, pred_mask = slide_partial_label(y_batch_train, batch_idxs, batch, num_labels)
            # make each row a valid probability distribution
            nz = tf.reshape(tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32), [batch, 1])
            y_true = y_true / nz

            # set non-active neuron true values to 0 because shouldn't count against SLIDE
            y_true = tf.math.multiply(pred_mask, y_true)

            # set non-active weights to zero so their gradients are small (~will try to become fully accurate later)
            final_layer = model.layers[-1].get_weights()
            final_layer[0][:, non_active_idx] = 0
            final_layer[1][non_active_idx] = 0
            model.layers[-1].set_weights(final_layer)

            # perform gradient update
            with tf.GradientTape() as tape:
                y_pred = Method.model(x_batch_train, training=True)
                y_pred = tf.math.multiply(pred_mask, y_pred)
                loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

            # apply backpropagation after setting non-active weights to zero
            grads = tape.gradient(loss_value, model.trainable_weights)
            # update weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # compute accuracy (top 1) and loss for the minibatch
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)
            # accuracy calc
            acc1_metric.update_state(y_pred, y_true)
            acc1 = acc1_metric.result().numpy()
            top1.update(acc1, batch)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - record_time

            # store and save accuracy and loss values
            recorder.add_new(comp_time, comp_time, 0, lsh_time, acc1, test_acc,
                             loss_value.numpy(), top1.avg, losses.avg)
            recorder.save_to_file()

            # update model and reset neurons that were incorrectly backpropped
            Method.update(np.unique(concat_idx))

            # log every X batches
            total_batches += batch
            acc1_metric.reset_state()
            if iterations % 5 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, LSH Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, [%d Total Samples]"
                    % (rank, iterations, comp_time, lsh_time, loss_value.numpy(), acc1, total_batches)
                )

            iterations += 1

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()


def regular_train(rank, size, Method, optimizer, train_data, test_data, losses, top1, recorder, args, num_labels):

    # parameters
    total_batches = 0
    test_acc = np.NaN
    acc1_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    test_acc1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    lsh_time = 0
    comm_time = 0
    iterations = 1

    # update indices with new current index
    Method.ci = np.arange(num_labels)
    Method.bias_idx = Method.ci + Method.bias_start
    # get model
    model = Method.model

    for epoch in range(1, args.epochs+1):
        print("\nStart of epoch %d" % (epoch,))

        # shuffle training data each epoch
        train_data.shuffle(len(train_data))

        # iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in train_data:

            # compute test accuracy every X steps
            if iterations % args.steps_per_test == 0:
                if rank == 0:
                    for (x_batch_test, y_batch_test) in test_data:
                        y_pred_test = model(x_batch_test, training=False)
                        test_acc1.update_state(y_pred_test, tf.sparse.to_dense(y_batch_test))
                        test_acc = test_acc1.result().numpy()
                    print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                    recorder.add_testacc(test_acc)
                    test_acc1.reset_state()

            init_time = time.time()

            # communicate models amongst devices (if multiple devices are present)
            if size > 1:
                model, comm_time = Method.communicate(model, smart=False)

            # transform sparse label to dense sub-label
            batch = x_batch_train.get_shape()[0]
            y_true = tf.sparse.to_dense(y_batch_train)
            # make each row a valid probability distribution
            nz = tf.reshape(tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32), [batch, 1])
            y_true = y_true / nz

            # perform gradient update
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

            # apply backpropagation after setting non-active weights to zero
            grads = tape.gradient(loss_value, model.trainable_weights)
            # update weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # compute accuracy (top 1) and loss for the minibatch
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)
            # accuracy calc
            acc1_metric.update_state(y_pred, y_true)
            acc1 = acc1_metric.result().numpy()
            top1.update(acc1, batch)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - record_time

            # store and save accuracy and loss values
            recorder.add_new((comp_time + comm_time), comp_time, comm_time, lsh_time, acc1, test_acc,
                             loss_value.numpy(), top1.avg, losses.avg)
            recorder.save_to_file()

            # log every X batches
            total_batches += batch
            acc1_metric.reset_state()
            if iterations % 5 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, [%d Total Samples]"
                    % (rank, iterations, (comp_time+comm_time), loss_value.numpy(), acc1, total_batches)
                )

            iterations += 1

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
