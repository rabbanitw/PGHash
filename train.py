import tensorflow as tf
import numpy as np
from sparse_bce import sparse_bce, sparse_bce_lsh
from accuracy import compute_accuracy, AverageMeter
from unpack import update_full_model, get_sub_model, get_full_dense
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
import time


def run_lsh(model, data, final_dense_w, sdim, num_tables, cr):

    # get input layer for LSH
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output,
    )
    in_layer = feature_extractor(data).numpy()

    # run LSH to find the most important weights
    return pg_avg(in_layer, final_dense_w, sdim, num_tables, cr)


def train(model, optimizer, communicator, train_data, test_data, full_model, epochs, sdim=8, num_tables=50,
          cr=0.1, steps_per_lsh=3, lsh=True):

    top1 = AverageMeter()
    top5 = AverageMeter()
    total_batches = 0
    start_idx_b = full_model.size - 670091
    used_idx = None
    cur_idx = None

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):

            batch, s = x_batch_train.get_shape()

            # run lsh here before training if being used
            if lsh:
                # periodic lsh
                if step % steps_per_lsh == 0:

                    # compute LSH
                    final_dense = get_full_dense(full_model)
                    cur_idx = run_lsh(model, x_batch_train, final_dense, sdim, int(num_tables/10), cr)

                    # receive sub-model corresponding to the outputted indices
                    w, bias = get_sub_model(full_model, cur_idx, start_idx_b)
                    weights = model.get_weights()
                    weights[-2] = w
                    weights[-1] = bias

                    # set the new model weight
                    model.set_weights(weights)

            t = time.time()
            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.
                y_pred = model(x_batch_train, training=True)

                # y_pred2 = np.zeros((batch, 670091))
                # y_pred2[:, cur_idx] = y_pred.numpy()
                # y_pred = tf.convert_to_tensor(y_pred2, dtype=tf.float32)
                # print(cur_idx[0:5])
                # print(np.linalg.norm(y_pred2[:, cur_idx[0]] - y_pred[:, 0]))

                # Compute the loss value for this minibatch.
                if lsh:
                    loss_value = sparse_bce_lsh(y_batch_train, y_pred, cur_idx)
                else:
                    loss_value = sparse_bce(y_batch_train, y_pred)

                # compute accuracy for the minibatch (top 1 and 5)
                acc1 = compute_accuracy(y_batch_train, y_pred, topk=1)
                acc5 = compute_accuracy(y_batch_train, y_pred, topk=5)
                top1.update(acc1, batch)
                top5.update(acc5, batch)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # communication happens here
            if lsh:
                comm_start = time.time()
                full_model, d_comm_time = communicator.communicate(model, full_model, cur_idx, start_idx_b)
                comm_t = time.time() - comm_start
            else:
                comm_start = time.time()
                d_comm_time = communicator.communicate(model)
                comm_t = time.time() - comm_start

            #print('Step Finished in %f Seconds With %f Step Accuracy and %f Epoch Accuracy'
            #      % ((time.time() - t), acc1, top1.avg))

            total_batches += batch
            # Log every 200 batches.
            if step % 5 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % total_batches)

        # reset accuracy statistics for next epoch
        top1.reset()
