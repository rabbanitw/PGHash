import tensorflow as tf
import numpy as np
import time
from misc import compute_accuracy_lsh


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
    return tf.convert_to_tensor(y_true[:, sub_idx], dtype=tf.float32)


def pg_train(rank, PGHash, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args, num_labels):

    # parameters
    cur_idx = None
    total_batches = 0
    iterations = 0
    test_acc = np.NaN

    # get model
    model = PGHash.return_model()

    for epoch in range(args.epochs):
        print("\nStart of epoch %d" % (epoch,))

        # iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in train_data:

            if args.lsh:
                lsh_init = time.time()
                # update full model
                PGHash.update_full_model(model)
                # compute LSH
                cur_idx = PGHash.lsh_avg_simple(x_batch_train)
                # get new model
                model = PGHash.get_new_model()
                lsh_time = time.time() - lsh_init
            else:
                lsh_time = 0

            for sub_batch in range(args.q):
                x = x_batch_train[(sub_batch * args.train_bs):((sub_batch + 1) * args.train_bs), :]
                y = y_batch_train[(sub_batch * args.train_bs):((sub_batch + 1) * args.train_bs), :]

                init_time = time.time()

                # communicate models amongst devices
                model, comm_time = PGHash.communicate(model)

                # transform sparse label to dense sub-label
                batch = x.get_shape()[0]
                y_true = get_partial_label(y, cur_idx, batch, num_labels)

                # perform gradient update
                with tf.GradientTape() as tape:
                    y_pred = model(x, training=True)
                    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
                grads = tape.gradient(loss_value, model.trainable_weights)
                optimizer.apply_gradients(zip(grads, model.trainable_weights))

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
                if iterations % 10 == 0:
                    print(
                        "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, [%d Total Samples]"
                        % (rank, iterations, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                    )

                # compute test accuracy every X steps
                if iterations % args.steps_per_test == 0:
                    if rank == 0:
                        PGHash.update_full_model(model)
                        test_acc = PGHash.test_full_model(test_data, test_top1)
                        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations, test_acc))
                        recorder.add_testacc(test_acc)
                        test_top1.reset()

                iterations += 1

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()

