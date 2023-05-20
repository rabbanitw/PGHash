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


class SampledSoftmaxLoss:
    """
    Class that instantiates a sampled softmax loss
    ...
    Attributes
    ----------
    num_classes : int
        Total number of classes in the softmax output
    sample_frac : float
        Fraction of num_classes to sample
    num_true_labels: int
        Number of true labels in the final output (1 for multiclass classfication)
    loss_name: string
        (Optional) Name for the loss function
    Methods
    -------
    loss(y_true, data):
        Returns the sampled softmax loss given the true label and the all additional data
    """

    def __init__(self, num_classes, num_sampled, num_true_labels=1, loss_name="sampled_softmax_loss"):
        self.num_classes = num_classes
        self.num_true_labels = num_true_labels
        self.num_sampled = num_sampled
        self.loss_name = loss_name

    def loss(self, y_true, data):
        """
        inputs to the softmax layer
        """
        inp, bias, weights = data
        if self.num_true_labels == 1:
            labels = tf.expand_dims(y_true, -1)
        else:
            labels = y_true

        logits = tf.matmul(inp, weights)
        logits = tf.nn.bias_add(logits, bias)

        return logits, tf.reduce_mean(tf.nn.sampled_softmax_loss(
            weights=tf.transpose(weights),
            biases=bias,
            labels=labels,
            inputs=inp,
            num_true=self.num_true_labels,
            num_sampled=self.num_sampled,
            num_classes=self.num_classes,
            remove_accidental_hits=True,
            name=self.loss_name
        ))


class SampledSoftmax(tf.keras.Model):

    def __init__(self, layer_dims, num_sampled):
        super().__init__()

        self.input_size = layer_dims[0]
        self.hls = layer_dims[1]
        self.num_classes = layer_dims[2]

        inp = tf.keras.layers.Input(shape=(self.input_size,), sparse=True)
        dense1 = tf.keras.layers.Dense(self.hls, activation="relu")(inp)
        output_layer = tf.keras.layers.Dense(self.num_classes)
        out = output_layer(dense1)

        self.full_model = tf.keras.Model(inp, out)
        self.half_model = tf.keras.Model(inp, dense1)

        self.out_weights = output_layer.weights[0]
        self.out_bias = output_layer.bias

        self.train_loss = SampledSoftmaxLoss(self.num_classes, num_sampled).loss
