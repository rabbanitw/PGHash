import tensorflow as tf
from mlp import SparseNeuralNetwork
from dataloader import load_amazon670
from bce_loss import sparse_bce
import time
import os


def train(train_data, test_data):

    layer_dims = [135909, 128, 670091]
    model = SparseNeuralNetwork(layer_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    epochs = 2
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):

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

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            print('Step Finished in %f Seconds' % (time.time() - t))

            # Log every 200 batches.
            if step % 5 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
                print("Seen so far: %s samples" % ((step + 1) * batch_size))


if __name__ == '__main__':
    batch_size = 128
    print('Loading data...')
    train_data, test_data = load_amazon670(batch_size)
    print('Beginning training...')
    train(train_data, test_data)
