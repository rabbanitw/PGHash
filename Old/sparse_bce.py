import tensorflow as tf
import numpy as np


def sparse_bce(y_true, y_pred, epsilon=1e-4):
    idx = y_true.indices
    one_loss = tf.math.reduce_sum(tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))))
    true_loss = 0
    for i in range(len(idx)):
        true_loss += (tf.math.log(tf.math.maximum(y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon)))
                      - tf.math.log(tf.math.maximum(1. - y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon))))
    return -(one_loss + tf.constant(true_loss)) / tf.cast(tf.size(y_pred), tf.float32)


def sparse_bce_lsh(y_true, y_pred, lsh_idx, epsilon=1e-4):

    idx = y_true.indices
    size_true = np.prod(y_true.get_shape())
    # size_pred = tf.size(y_pred)
    # one loss for values in compressed network
    one_loss = tf.math.reduce_sum(tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))))
    # one_loss += tf.cast((size_true-size_pred), tf.float32)*tf.math.log(0.5)
    true_loss = 0
    for i in range(len(idx)):
        index = idx[i][1].numpy()
        # add in correct loss depending upon if it is part of the compressed network
        if np.isin(index, lsh_idx):
            row = np.where(index == lsh_idx)[0][0]
            true_loss += (tf.math.log(tf.math.maximum(y_pred[idx[i][0], row], tf.constant(epsilon))) -
                          tf.math.log(tf.math.maximum(1. - y_pred[idx[i][0], row], tf.constant(epsilon))))
        # else:
        #    true_loss += tf.math.log(tf.constant(epsilon))
    return -(one_loss + tf.constant(true_loss)) / tf.cast(size_true, tf.float32)
