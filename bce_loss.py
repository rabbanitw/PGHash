import tensorflow as tf


def sparse_bce(y_true, y_pred, epsilon=1e-2):
    idx = y_true.indices
    false_mat = tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))).numpy()
    true_loss = 0
    for i in range(len(idx)):
        true_loss += tf.math.log(tf.math.maximum(y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon)))
        false_mat[idx[i][0], idx[i][1]] = 0
    return -(tf.math.reduce_mean(false_mat) + true_loss/len(idx))
