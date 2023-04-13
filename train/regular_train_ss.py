import numpy as np
import tensorflow as tf
import time


def regular_train_ss(rank, size, Method, optimizer, train_data, test_data, losses, top1, recorder, args):

    # parameters
    total_batches = 0
    test_acc = np.NaN
    lsh_time = 0
    comm_time = 0
    iterations = 1
    acc1 = 0
    num_final_layers = Method.nl*args.cr
    full_size = np.arange(Method.nl-1)

    # update indices with new current index
    Method.ci = np.arange(Method.nl)
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
                    test_data.shuffle(len(test_data))
                    sub_test_data = test_data.take(30)
                    total_test_acc = 0
                    examples = 0
                    for (x_batch_test, y_batch_test) in sub_test_data:
                        test_acc = model((x_batch_test, y_batch_test), training=False)
                        bs = x_batch_test.get_shape()[0]
                        total_test_acc += test_acc * bs
                        examples += bs
                    ta = total_test_acc/examples
                    print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, ta))
                    recorder.add_testacc(ta)

            init_time = time.time()

            # communicate models amongst devices (if multiple devices are present)
            if size > 1:
                model, comm_time = Method.communicate(model)

            # transform sparse label to dense sub-label
            batch = x_batch_train.get_shape()[0]

            '''
            y = tf.sparse.to_dense(y_batch_train).numpy()
            nz = np.count_nonzero(y, axis=1)
            max_num_labels = np.max(nz)
            # nz_indices = np.empty((batch, max_num_labels))
            nz_indices = np.ones((batch, max_num_labels)) * (Method.nl-1)
            for i in range(y.shape[0]):
                num_nz = nz[i]
                y_row = y[i, :]
                nz_indices[i, :num_nz] = full_size[y_row>0]
                # if num_nz < max_num_labels:
                #    nz_indices[i, num_nz:] = nz_indices[i, num_nz-1]
            y_labels = tf.convert_to_tensor(nz_indices)
            '''

            # perform gradient update
            with tf.GradientTape() as tape:
                loss_value = model([x_batch_train, y_batch_train], training=True)
                # loss_value = model([x_batch_train, y], training=True)

            # apply backpropagation after setting non-active weights to zero
            grads = tape.gradient(loss_value, model.trainable_weights)
            # update weights
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            # compute accuracy (top 1) and loss for the minibatch
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)
            # accuracy calc
            top1.update(acc1, batch)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - record_time

            # store and save accuracy and loss values
            recorder.add_new((comp_time + comm_time), comp_time, comm_time, lsh_time, acc1, test_acc,
                             loss_value.numpy(), top1.avg, losses.avg, Method.nl, num_final_layers)
            recorder.save_to_file()

            # log every X batches
            total_batches += batch
            if iterations % 5 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, [%d Total Samples]"
                    % (rank, iterations, (comp_time+comm_time), loss_value.numpy(), total_batches)
                )

            iterations += 1

        total_test_acc = 0
        examples = 0
        for (x_batch_test, y_batch_test) in test_data:
            test_acc = model((x_batch_test, y_batch_test), training=False)
            bs = x_batch_test.get_shape()[0]
            total_test_acc += test_acc * bs
            examples += bs
        ta = total_test_acc / examples
        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations - 1, ta))
        recorder.add_testacc(ta)
        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
