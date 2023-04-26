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
    num_final_layers = Method.nl*args.cr
    acc1_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
    test_acc1 = tf.keras.metrics.TopKCategoricalAccuracy(k=1)

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
                    for (x_batch_test, y_batch_test) in sub_test_data:
                        y_pred_test = model.full_model(x_batch_test, training=False)
                        test_acc1.update_state(y_pred_test, tf.sparse.to_dense(y_batch_test))
                    test_acc = test_acc1.result().numpy()
                    print("Step %d: Top 1 Test Accuracy %.4f" % (iterations - 1, test_acc))
                    recorder.add_testacc(test_acc)
                    test_acc1.reset_state()

            init_time = time.time()

            # communicate models amongst devices (if multiple devices are present)
            if size > 1:
                model, comm_time = Method.communicate(model)

            # transform sparse label to dense sub-label
            batch = x_batch_train.get_shape()[0]

            y_true = tf.sparse.to_dense(y_batch_train)

            # randomize the label selected (since there's a tie)
            y_true_ind = tf.math.argmax(y_true*tf.random.uniform(tf.shape(y_true)), axis=1)
            # y_true_ind = tf.math.argmax(y_true, axis=1)

            # perform gradient update
            with tf.GradientTape() as tape:
                inp = model.half_model(x_batch_train, training=True)  # Forward pass
                y_pred, loss_value = model.train_loss(y_true_ind, [inp, model.out_bias, model.out_weights])

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
                             loss_value.numpy(), top1.avg, losses.avg, Method.nl, num_final_layers)
            recorder.save_to_file()

            # log every X batches
            acc1_metric.reset_state()
            total_batches += batch
            if iterations % 5 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, [%d Total Samples]"
                    % (rank, iterations, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                )

            iterations += 1

        if rank == 0:
            for (x_batch_test, y_batch_test) in test_data:
                y_pred_test = model.full_model(x_batch_test, training=False)
                test_acc1.update_state(y_pred_test, tf.sparse.to_dense(y_batch_test))
            test_acc = test_acc1.result().numpy()
            print("Step %d: Top 1 Test Accuracy %.4f" % (iterations - 1, test_acc))
            recorder.add_testacc(test_acc)
            test_acc1.reset_state()

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()