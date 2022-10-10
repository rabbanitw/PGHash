import tensorflow as tf
import numpy as np
from sparse_bce import sparse_bce
from accuracy import compute_accuracy, AverageMeter
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
import time


def run_lsh(model, data, sdim, num_tables, cr):

    # get input layer for LSH
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output,
    )
    in_layer = feature_extractor(data).numpy()

    # get the final dense layer to be compressed
    weights = model.get_weights()
    final_dense = weights[-2]

    # run LSH to find the most important weights
    max_idx = pg_avg(in_layer, final_dense, sdim, num_tables, cr)

    # zero out all other weights that are unimportant
    new_final_dense = np.zeros_like(final_dense)
    new_final_dense[:, max_idx] = final_dense[:, max_idx]
    weights[-2] = new_final_dense

    # set the new model weight
    model.set_weights(weights)

    return max_idx, final_dense


def train(model, optimizer, communicator, train_data, test_data, epochs, sdim=8, num_tables=50, cr=0.1,
          steps_per_lsh=3, lsh=True):

    top1 = AverageMeter()
    top5 = AverageMeter()
    total_batches = 0
    used_idx = None
    cur_idx = None
    final_dense = None

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):

            b, s = x_batch_train.get_shape()

            # run lsh here before training if being used
            if lsh:
                # periodic lsh
                if step % steps_per_lsh == 0:
                    cur_idx, final_dense = run_lsh(model, x_batch_train, sdim, num_tables, cr)
                else:
                    weights = model.get_weights()
                    final_dense = weights[-2]
                    new_final_dense = np.zeros_like(final_dense)
                    new_final_dense[:, cur_idx] = final_dense[:, cur_idx]
                    weights[-2] = new_final_dense
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

                # Compute the loss value for this minibatch.
                loss_value = sparse_bce(y_batch_train, y_pred)

                # compute accuracy for the minibatch (top 1 and 5)
                acc1 = compute_accuracy(y_batch_train, y_pred, topk=1)
                acc5 = compute_accuracy(y_batch_train, y_pred, topk=5)
                top1.update(acc1, b)
                top5.update(acc5, b)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # communication happens here
            if lsh:
                weights = model.get_weights()
                final_dense[:, cur_idx] = weights[-2][:, cur_idx]
                weights[-2] = final_dense
                model.set_weights(weights)

            comm_start = time.time()
            d_comm_time = communicator.communicate(model)
            comm_t = time.time() - comm_start

            #print('Step Finished in %f Seconds With %f Step Accuracy and %f Epoch Accuracy'
            #      % ((time.time() - t), acc1, top1.avg))

            total_batches += b
            # Log every 200 batches.
            if step % 5 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % total_batches)

        # reset accuracy statistics for next epoch
        top1.reset()
