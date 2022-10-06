import tensorflow as tf
# import tensorflow_datasets as tfds
from mlp import SparseNeuralNetwork
from dataloader import load_amazon670
from sparse_bce import sparse_bce
from accuracy import compute_accuracy
import time


if __name__ == '__main__':
    a = tf.random.uniform(shape=(6, 6))
    b = tf.constant([[1, 0, 0, 0.9, 0.8, 0], [0.7, 1, 0.3, 0.9, 0.5, 0], [1, 0, 0, 0, 0.6, 0.7], [1, 0, 0, 0, 0.6, 0.7],
                     [0, 0, 0.7, 0.8, 1, 0], [0.6, 0, 0, 0.8, 0, 1]])
    c = tf.sparse.eye(6)

    print(b)

    acc = compute_accuracy(c, b, topk=2)
    print(acc)

