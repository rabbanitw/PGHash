import tensorflow as tf
import numpy as np
# import tensorflow_datasets as tfds
from mlp import SparseNeuralNetwork
from dataloader import load_amazon670
from sparse_bce import sparse_bce, sparse_bce_lsh
from misc import compute_accuracy, compute_accuracy_lsh
from dataloader import load_amazon670
import time


if __name__ == '__main__':

    '''
    #b = tf.constant([[1, 0, 0, 0.9, 0.8, 0], [0.7, 1, 0.3, 0.9, 0.5, 0], [1, 0, 0, 0, 0.6, 0.7], [1, 0, 0, 0, 0.6, 0.7],
    #                 [0, 0, 0.7, 0.8, 1, 0], [0.6, 0, 0, 0.8, 0, 1]])
    c = tf.sparse.eye(5)
    # b = tf.constant([[1., 0., 0., 0., 0., 0.], [0., 1., 0., 0., 0., 0.], [0., 0., 1., 0., 0., 0.], [0., 0., 0., 1., 0., 0.],
    #                 [0., 0., 0., 0., 1., 0.], [0., 0., 0., 0., 0., 1.]])

    b = tf.constant(
        [[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.], [0., 0., 1., 0., 0.], [0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1.], [0., 0., 0., 0., 1.]])

    b = tf.random.uniform(shape=(5, 5))
    print(b)
    # d = tf.sparse.from_dense(tf.random.uniform(shape=(10, 10)))
    lsh_idx = np.array([0, 1, 2, 3, 4])


    acc = compute_accuracy_lsh(c, b, lsh_idx, topk=1)
    print(acc)
    # loss = sparse_bce(c, b)
    # print(loss)
    # loss = sparse_bce_lsh(c, b, np.arange(50))
    # print(loss)
    '''
    #'''
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    tr, te = load_amazon670(0, 1, 256)
    for (x, y) in tr:
        t = time.time()
        yy = tf.sparse.to_dense(y)
        print(time.time() - t)
        # acc = compute_accuracy_lsh(y, tf.sparse.to_dense(y), np.arange(y.shape[1]), topk=5)
        loss = bce(yy, yy)
        #acc2 = compute_accuracy(yy, yy, topk=5)
        print(loss)
        print(time.time()-t)
        break
    a = 0
    #'''

