import tensorflow as tf
# import tensorflow_datasets as tfds
from mlp import SparseNeuralNetwork
from dataloader import load_amazon670
from bce_loss import custom_bce
import os




def train():
    layer_dims = [100, 128, 10]
    nn = SparseNeuralNetwork(layer_dims)

    sparse_data = tf.sparse.SparseTensor(
        indices=[(0, 0), (0, 1), (0, 2),
                 (4, 3), (5, 0), (5, 1)],
        values=[1, 1, 1, 1, 1, 1],
        dense_shape=(10, 100)
    )

    sparse_y = tf.sparse.SparseTensor(
        indices=[(0, 0), (0, 1), (0, 2),
                 (4, 3), (5, 0), (5, 1)],
        values=[1., 1., 1., 1., 1., 1.],
        dense_shape=(10, 10),
    )

    a = nn(sparse_data)
    loss = custom_bce(sparse_y, a)
    print(loss)



if __name__ == '__main__':
    # load_amazon670()
    train()
