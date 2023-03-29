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


class SparseLinear(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        super(SparseLinear, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, output_dim), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(output_dim,), dtype="float32"), trainable=True
        )

    def call(self, inputs):
        input, mask = inputs
        weight = tf.sparse.from_dense(tf.multiply(self.w, tf.squeeze(mask)))
        return tf.sparse.sparse_dense_matmul(input, weight) + self.b


def SparseNeuralNetwork2(layer_dims):
    inputA = tf.keras.layers.Input(shape=(layer_dims[0],), sparse=True)
    inputB = tf.keras.layers.Input(shape=(layer_dims[1], layer_dims[2]))
    x = DenseLayer(layer_dims[1])(inputA)
    x = tf.keras.activations.relu(x)
    x = SparseLinear(layer_dims[1], layer_dims[2])([x, inputB])
    # create the model
    model = tf.keras.Model(inputs=[inputA, inputB], outputs=x)
    # return the constructed network architecture
    return model

