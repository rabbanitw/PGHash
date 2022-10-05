import tensorflow as tf


def sparse_bce(y_true, y_pred, epsilon=1e-4):
    idx = y_true.indices
    loss_mat = tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))).numpy()
    for i in range(len(idx)):
        loss_mat[idx[i][0], idx[i][1]] = \
            tf.math.log(tf.math.maximum(y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon)))
    return -tf.math.reduce_mean(loss_mat)
