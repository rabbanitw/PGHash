import tensorflow as tf
import numpy as np


class SS_Linear(tf.keras.layers.Layer):
    def __init__(self, units, input_dim, num_classes, cr):
        super().__init__()
        self.num_classes = num_classes
        self.num_sampled = int(num_classes * cr)
        self.acc = tf.keras.metrics.TopKCategoricalAccuracy(k=1)
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"),
            trainable=True,
        )
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
        )

    def call(self, inputs, training=True):
        (x, y) = inputs
        if training:

            y = tf.sparse.to_dense(y)
            # randomize the label selected (since there's a tie)
            # label_idx = tf.expand_dims(tf.math.argmax(y*tf.random.uniform(tf.shape(y)), axis=1), axis=1)
            # regular argmax
            label_idx = tf.expand_dims(tf.math.argmax(y, axis=1), axis=1)
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=tf.transpose(self.w), biases=self.b, labels=label_idx,
                                                             inputs=x, num_sampled=self.num_sampled,
                                                             num_classes=self.num_classes,
                                                             #num_true=y.get_shape()[1],
                                                             remove_accidental_hits=False))
            return loss
        else:
            logits = tf.matmul(x, self.w) + self.b
            logits = tf.convert_to_tensor(logits.numpy()[:, :-1])
            y = tf.sparse.to_dense(y)
            self.acc.update_state(logits, y)
            accuracy = self.acc.result().numpy()
            self.acc.reset_state()
            return accuracy

class Sparse_Linear(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Sparse_Linear, self).__init__()
        self.w = self.add_weight(
            shape=(input_dim, units), initializer="random_normal", trainable=True
        )
        self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.sparse.sparse_dense_matmul(inputs, self.w) + self.b


class SampledSoftmax(tf.keras.Model):

    def __init__(self, layer_dims, cr=0.1, name="sampled_softmax", **kwargs):
        super(SampledSoftmax, self).__init__(name=name, **kwargs)
        self.sparse_dense = Sparse_Linear(layer_dims[1], layer_dims[0])
        self.sampled_softmax_layer = SS_Linear(layer_dims[2], layer_dims[1], layer_dims[2], cr)

    def call(self, inputs, training=True):
        (x, y) = inputs
        output = self.sparse_dense(x)
        return self.sampled_softmax_layer([output, y], training=training)


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

