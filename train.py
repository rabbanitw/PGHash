import tensorflow as tf
from mlp import SparseNeuralNetwork
from sparse_bce import sparse_bce
from accuracy import compute_accuracy, AverageMeter
import time


def train(train_data, test_data, epochs):

    layer_dims = [135909, 128, 670091]
    model = SparseNeuralNetwork(layer_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    top1 = AverageMeter()
    total_batches = 0

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):

            b, s = x_batch_train.get_shape()

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

                # compute accuracy for the minibatch
                acc = compute_accuracy(y_batch_train, y_pred, topk=1)
                top1.update(acc, b)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print('Step Finished in %f Seconds With %f Step Accuracy and %f Epoch Accuracy'
                  % ((time.time() - t), acc, top1.avg))

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
