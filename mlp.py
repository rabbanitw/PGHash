import tensorflow as tf


def DenseLayer(output_dim):
    return tf.keras.layers.Dense(output_dim)


def SparseNeuralNetwork(layer_dims, sparsity=True):
    inputs = tf.keras.layers.Input(shape=(layer_dims[0],), sparse=sparsity)
    x = DenseLayer(layer_dims[1])(inputs)
    x = tf.keras.activations.relu(x)
    x = DenseLayer(layer_dims[2])(x)
    # create the model
    model = tf.keras.Model(inputs, x)
    # return the constructed network architecture
    return model
