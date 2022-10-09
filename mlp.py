import tensorflow as tf


def DenseLayer(output_dim):
    return tf.keras.layers.Dense(output_dim)


def SparseNeuralNetwork(layer_dims, sparsity=True):
    inputs = tf.keras.layers.Input(shape=(layer_dims[0],), sparse=sparsity)
    x = DenseLayer(layer_dims[1])(inputs)
    if len(layer_dims) > 2:
        for i in range(2, len(layer_dims)):
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.activations.relu(x)
            x = DenseLayer(layer_dims[i])(x)
    x = tf.keras.activations.sigmoid(x)
    # create the model
    model = tf.keras.Model(inputs, x)
    # return the constructed network architecture
    return model


'''
class NeuralNetwork(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        self.input_dims = input_dims
        self.output_dims = output_dims
        super(NeuralNetwork, self).__init__()
        self.layers = []
        for i in range(len(self.input_dims)):
            self.layers.append(tf.keras.layers.Dense(self.output_dims[i], input_shape=(self.input_dims[i],),
                                                     activation='relu'))

    def call(self, inputs):
        for i in range(len(self.input_dims)):
            inputs = self.layers[i](inputs)
        return inputs


class SparseNeuralNetwork(tf.keras.Model):
    def __init__(self, input_dims, output_dims):
        self.input_dims = input_dims
        self.output_dims = output_dims
        super(SparseNeuralNetwork, self).__init__()
        self.layers = []
        for i in range(len(self.input_dims)):
            # self.layers.append(tf.keras.layers.Dense(self.output_dims[i], input_shape=(self.input_dims[i],), , activation='relu'))
            self.layers.append(Dense(self.output_dims[i], input_shape=(self.input_dims[i],), activation='relu'))

    def call(self, inputs):
        for i in range(len(self.input_dims)):
            inputs = self.layers[i](inputs)
        return inputs
'''
