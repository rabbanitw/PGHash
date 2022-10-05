import tensorflow as tf
# import tensorflow_datasets as tfds
from mlp import SparseNeuralNetwork
from dataloader import load_amazon670
from bce_loss import sparse_bce
import time


if __name__ == '__main__':
    a = tf.random.uniform(shape=(5, 5))
    b = tf.eye(5)
    c = tf.sparse.eye(5)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    print(bce(b, a).numpy())

    print(sparse_bce(c, a))

