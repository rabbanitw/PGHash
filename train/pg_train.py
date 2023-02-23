import numpy as np
import tensorflow as tf
import time
from util.misc import compute_accuracy_lsh


def pg_train(rank, size, Method, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args):

    # parameters
    total_batches = 0
    iterations = 1
    test_acc = np.NaN
    comm_time = 0
    if args.cr == 1:
        smartavg = False
    else:
        smartavg = True

    num_labels = Method.nl
    num_features = Method.nf

    for epoch in range(1, args.epochs+1):
        print("\nStart of epoch %d" % (epoch,))

        # shuffle training data each epoch
        train_data.shuffle(len(train_data))

        # iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in train_data:

            # figure how many batches are in the mega batch Q
            batches_per_q = np.ceil(x_batch_train.shape[0] / args.train_bs).astype(np.int32)

            # update full model
            Method.update_full_model(Method.model)

            # compute LSH
            lsh_init = time.time()
            # active_idx, sample_active_idx = Method.lsh_hamming(Method.model, x_batch_train)
            active_idx, sample_active_idx = Method.lsh_hamming_opt(Method.model, x_batch_train)
            lsh_time = time.time() - lsh_init

            if size > 1:
                # send indices to root (server)
                Method.exchange_idx()

            # update model
            Method.update_model()

            for sub_batch_idx in range(batches_per_q):

                # compute test accuracy every X steps
                if iterations % args.steps_per_test == 0:
                    if rank == 0:
                        Method.update_full_model(Method.model)
                        test_acc = Method.test_full_model(test_data, test_top1)
                        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                        recorder.add_testacc(test_acc)
                        test_top1.reset()

                x = tf.sparse.slice(x_batch_train, start=[sub_batch_idx * args.train_bs, 0],
                                    size=[args.train_bs, num_features])
                y = tf.sparse.slice(y_batch_train, start=[sub_batch_idx * args.train_bs, 0],
                                    size=[args.train_bs, num_labels])

                # compute batch size
                batch_size = x.get_shape()[0]

                # don't count LSH time towards subsequent batches of the mega batch_size
                if sub_batch_idx == 1:
                    lsh_time = 0

                init_time = time.time()

                # communicate models amongst devices (if multiple devices are present)
                if size > 1:
                    model, comm_time = Method.communicate(Method.model, smart=smartavg)

                # preprocess true label
                y_true = tf.sparse.to_dense(y)

                # make sure all samples are divided by number of labels
                nz = tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
                y_true = y_true / nz

                '''
                # TRY TO CREATE MASK FROM INDICES WITHOUT HAVING TO USE FORLOOP (SAVE 0.025 seconds)
                mask = np.zeros((batch_size, num_labels))
                for j in range(batch_size):
                    mask[j, sample_active_idx[j + sub_batch_idx * args.train_bs]] = 1
                mask = mask[:, active_idx]
                active_mask = tf.convert_to_tensor(mask, dtype=tf.dtypes.float32)
                softmax_mask = tf.where(active_mask == 1, 0., tf.float32.min / 10)
                '''

                # shorten the true label
                y_true = tf.gather(y_true, indices=active_idx, axis=1)

                # perform gradient update
                with tf.GradientTape() as tape:
                    # This is custom using only ACTIVE neurons as part of sum
                    y_pred = Method.model(x)
                    #y_pred = tf.math.add(y_pred, softmax_mask)
                    log_sm = tf.nn.log_softmax(y_pred, axis=1)
                    # zero out non-active neurons for each sample
                    # log_sm = tf.math.multiply(log_sm, active_mask)
                    loss_value = -tf.reduce_mean(tf.reduce_sum(tf.math.multiply(log_sm, y_true), axis=1, keepdims=True))

                grads = tape.gradient(loss_value, Method.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, Method.model.trainable_weights))

                # compute accuracy (top 1) and loss for the minibatch
                rec_init = time.time()
                losses.update(np.array(loss_value), batch_size)
                acc1 = compute_accuracy_lsh(y_pred, y, active_idx, num_labels)
                top1.update(acc1, batch_size)
                record_time = time.time() - rec_init
                comp_time = (time.time() - init_time) - (record_time + comm_time)

                if sub_batch_idx != 0:
                    lsh_time = 0

                # store and save accuracy and loss values
                recorder.add_new(comp_time + comm_time + lsh_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                                 loss_value.numpy(), top1.avg, losses.avg)
                recorder.save_to_file()

                # log every X batches
                total_batches += batch_size
                if iterations % 5 == 0:
                    print(
                        "(Rank %d) Step %d: Epoch Time %f, Comm Time %f, LSH Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, "
                        "[%d Total Samples]" % (rank, iterations, (comp_time + comm_time), comm_time, lsh_time,
                                                loss_value.numpy(), acc1, total_batches)
                    )
                iterations += 1

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
