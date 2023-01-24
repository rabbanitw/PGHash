import numpy as np
import tensorflow as tf
import time
from misc import compute_accuracy_lsh
import copy


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

    # true_idx = sparse_y.indices.numpy()
    # y_true = np.zeros((batch_size, full_num_labels))
    # for i in true_idx:
    #    y_true[i[0], i[1]] = 1

    y_true = tf.sparse.to_dense(sparse_y).numpy()

    # maybe divide by num_labels here first...
    nz = np.count_nonzero(y_true, axis=1).reshape(batch_size, 1)
    nz[nz == 0] = 1
    y_true = y_true / nz

    y_true = y_true[:, sub_idx]
    #nz = np.count_nonzero(y_true, axis=1).reshape(batch_size, 1)
    #nz[nz == 0] = 1
    #y_true = y_true / nz
    return tf.convert_to_tensor(y_true, dtype=tf.float32)


def get_partial_label_mask(sparse_y, sub_idx, sample_idx, batch_size):
    '''
    Takes a sparse full label and converts it into a dense sub-label corresponding to the output nodes in the
    sub-architecture for a given device
    :param sparse_y: Sparse full labels
    :param sub_idx: Indices for which output nodes are used/activated in the sub-architecture
    :param batch_size: Batch size
    :param full_num_labels: Total number of output nodes
    :return: Dense sub-label corresponding to given output nodes
    '''

    y_true = tf.sparse.to_dense(sparse_y).numpy()

    # make sure all samples are divided by number of labels
    nz = np.count_nonzero(y_true, axis=1, keepdims=True)
    y_true = y_true / nz

    mask = np.zeros((batch_size, y_true.shape[1]))
    for j in range(batch_size):
        mask[j, sample_idx[j]] = 1

    # mask the true label
    y_true = y_true * mask

    # shorten the true label and mask
    y_true = y_true[:, sub_idx]
    mask = mask[:, sub_idx]

    # make sure all samples are divided by number of labels (MAYBE DO THIS BEFORE!!)
    leftout_labels = nz - np.count_nonzero(y_true, axis=1, keepdims=True)
    # nz[nz == 0] = 1
    # y_true = y_true / nz

    return tf.convert_to_tensor(y_true, dtype=tf.float32), tf.convert_to_tensor(mask, dtype=tf.float32), \
           tf.convert_to_tensor(leftout_labels, dtype=tf.float32), tf.convert_to_tensor(1/nz, dtype=tf.float32)


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

    num_diff = tf.constant(num_labels - Method.num_c_layers, dtype=tf.float32)

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

            # if iterations % args.steps_per_lsh == 0 or iterations == 1:
            #    Method.lsh_tables()

            # cur_idx, per_sample_idx = Method.lsh(Method.model, x_batch_train)
            # cur_idx, per_sample_idx = Method.lsh_initial(Method.model, x_batch_train)
            # cur_idx, per_sample_idx = Method.lsh_vanilla(Method.model, x_batch_train)
            cur_idx, per_sample_idx = Method.lsh_hamming(Method.model, x_batch_train)
            if size > 1:
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
                # y_true = get_partial_label(y, cur_idx, batch, num_labels)
                y_true, pred_mask, leftover, label_frac = get_partial_label_mask(y, cur_idx, per_sample_idx, batch)

                # perform gradient update
                with tf.GradientTape() as tape:
                    y_pred = Method.model(x)
                    # y_pred = tf.math.multiply(pred_mask, y_pred)
                    # loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

                    max_logit = tf.math.maximum(tf.math.reduce_max(y_pred, axis=1, keepdims=True), 0)
                    e_logit = tf.math.maximum(tf.math.exp(y_pred - max_logit), 1e-7)
                    outside_e_logit = tf.math.maximum(tf.math.exp(0 - max_logit), 1e-7)
                    e_sum = (num_diff*outside_e_logit + tf.math.reduce_sum(e_logit, axis=1, keepdims=True))
                    log_sm = tf.math.log(e_logit / e_sum)

                    inner = tf.reduce_sum(tf.math.multiply(log_sm, y_true), axis=1, keepdims=True)
                    outer = leftover*label_frac*tf.math.log(outside_e_logit / e_sum)
                    loss_value = -tf.reduce_mean(inner + outer)


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
