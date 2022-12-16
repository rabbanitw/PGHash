import torch.nn as nn
import torch


class NeuralNetwork(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size):
        super(NeuralNetwork, self).__init__()
        # self.flatten = nn.Flatten()
        self.layer = nn.Sequential(
            nn.Linear(input_layer_size, hidden_layer_size, dtype=torch.float64),
            # nn.BatchNorm1d(hidden_layer_size),
            # nn.BatchNorm1d(hidden_layer_size, momentum=0.99, eps=0.001),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, output_layer_size, dtype=torch.float64)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.flatten(x)
        logits = self.layer(x)
        return logits
        # return self.softmax(logits)


import tensorflow as tf


def DenseLayer(output_dim):
    return tf.keras.layers.Dense(output_dim)


def SparseNeuralNetwork(layer_dims, sparsity=True, training=True):
    inputs = tf.keras.layers.Input(shape=(layer_dims[0],), sparse=sparsity)
    x = DenseLayer(layer_dims[1])(inputs)
    if len(layer_dims) > 2:
        for i in range(2, len(layer_dims)):
            # x = tf.keras.layers.BatchNormalization()(x, training=training)
            x = tf.keras.activations.relu(x)
            x = DenseLayer(layer_dims[i])(x)
    # x = tf.keras.activations.softmax(x)
    # x = tf.keras.activations.sigmoid(x)
    # create the model
    model = tf.keras.Model(inputs, x)
    # return the constructed network architecture
    return model
