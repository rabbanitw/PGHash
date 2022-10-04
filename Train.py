import tensorflow as tf
# import tensorflow_datasets as tfds
from mlp import SparseNeuralNetwork
from dataloader import load_amazon670
import os




def train():
    layer_dims = [100, 128, 10]
    nn = SparseNeuralNetwork(layer_dims)
    print('hi')


if __name__ == '__main__':
    # load_amazon670()
    train()
