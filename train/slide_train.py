import numpy as np
import tensorflow as tf
import time
from util.misc import compute_accuracy_lsh


def slide_train(rank, Method, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args):

    # parameters
    total_batches = 0
    iterations = 1
    test_acc = np.NaN
    comm_time = 0
    num_labels = Method.nl
    steps_per_rehash = args.steps_per_lsh

    for epoch in range(1, args.epochs + 1):
        print("\nStart of epoch %d" % (epoch,))

        # shuffle training data each epoch
        train_data.shuffle(len(train_data))

        # iterate over the batches of the dataset.
        for (x, y) in train_data:

            # update full model
            Method.update_full_model(Method.model)

            # compute LSH
            lsh_init = time.time()
            if (iterations - 1) % steps_per_rehash == 0:
                if args.dwta:
                    Method.rehash_wta()
                else:
                    Method.rehash()
            if args.dwta:
                active_idx, sample_active_idx, true_neurons_bool, fake_n = Method.lsh_vanilla_wta(Method.model, x)
            else:
                active_idx, sample_active_idx, true_neurons_bool, fake_n = Method.lsh_vanilla(Method.model, x)
            lsh_time = time.time() - lsh_init

            # send indices to root (server)
            if Method.size > 1:
                comm_time1 = Method.exchange_idx_vanilla(true_neurons_bool)

            # update model
            Method.update_model()

            # compute test accuracy every X steps
            if iterations % args.steps_per_test == 0:
                if rank == 0:
                    Method.update_full_model(Method.model)
                    test_acc = Method.test_full_model(test_data, test_top1, epoch_test=False)
                    print("Step %d: Top 1 Test Accuracy %.4f" % (iterations - 1, test_acc))
                    recorder.add_testacc(test_acc)
                    test_top1.reset()

            # communicate models amongst devices (if multiple devices are present)
            if Method.size > 1:
                active_neurons = Method.full_size[true_neurons_bool]
                comm_time2 = Method.communicate(Method.model, active_neurons)
                comm_time = comm_time1 + comm_time2

            # compute batch size
            batch_size = x.get_shape()[0]

            # document total number of active neurons across the batch
            num_active_neurons = np.count_nonzero(true_neurons_bool)

            init_time = time.time()

            # preprocess true label
            y_true = tf.sparse.to_dense(y)

            # make sure all samples are divided by number of labels
            nz = tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
            y_true = y_true / nz

            # TRY TO CREATE MASK FROM INDICES WITHOUT HAVING TO USE FORLOOP (SAVE 0.025 seconds)
            mask = np.zeros((batch_size, num_labels))
            for j in range(batch_size):
                mask[j, sample_active_idx[j]] = 1
            mask = mask[:, active_idx]
            active_mask = tf.convert_to_tensor(mask, dtype=tf.dtypes.float32)
            nonactive_mask = tf.where(active_mask == 0, 1., 0.)
            softmax_mask = tf.where(active_mask == 1, 0., tf.float32.min)

            # shorten the true label
            y_true = tf.gather(y_true, indices=active_idx, axis=1)

            # perform gradient update
            with tf.GradientTape() as tape:
                # This is custom using only ACTIVE neurons as part of sum
                y_pred = Method.model(x)
                y_pred = tf.math.add(y_pred, softmax_mask)
                log_sm = tf.nn.log_softmax(y_pred, axis=1)
                # zero out non-active neurons for each sample
                log_sm = tf.math.multiply(log_sm, active_mask)
                smce = tf.math.multiply(log_sm, y_true)
                smce = tf.stop_gradient(nonactive_mask * smce) + active_mask * smce
                loss_value = -tf.reduce_mean(tf.reduce_sum(smce, axis=1, keepdims=True))

            grads = tape.gradient(loss_value, Method.model.trainable_weights)
            optimizer.apply_gradients(zip(grads, Method.model.trainable_weights))

            # compute accuracy (top 1) and loss for the minibatch
            rec_init = time.time()
            losses.update(np.array(loss_value), batch_size)
            acc1 = compute_accuracy_lsh(y_pred, y, active_idx, num_labels)
            top1.update(acc1, batch_size)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - (record_time + comm_time)

            # store and save accuracy and loss values
            recorder.add_new(comp_time + comm_time + lsh_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                             loss_value.numpy(), top1.avg, losses.avg, num_active_neurons)
            recorder.save_to_file()

            # log every X batches
            total_batches += batch_size
            if iterations % 5 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Comm Time %f, LSH Time %f, Loss %.6f, Top 1 Train Accuracy %.4f,"
                    " Total Active Neurons %d, [%d Total Samples]" % (rank, iterations, (comp_time + comm_time),
                                                                     comm_time, lsh_time, loss_value.numpy(), acc1,
                                                                     num_active_neurons, total_batches)
                )
            iterations += 1

        # compute end of epoch testing
        if rank == 0:
            Method.update_full_model(Method.model)
            test_acc = Method.test_full_model(test_data, test_top1, epoch_test=True)
            print("Epoch %d: Top 1 Test Accuracy %.4f" % (epoch, test_acc))
            recorder.add_testacc(test_acc)
            test_top1.reset()

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
