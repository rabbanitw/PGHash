import tensorflow as tf


def sparse_bce(y_true, y_pred, epsilon=1e-4):
    idx = y_true.indices
    one_loss = tf.math.reduce_sum(tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))))
    true_loss = 0
    for i in range(len(idx)):
        true_loss += (tf.math.log(tf.math.maximum(y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon)))
                      - tf.math.log(tf.math.maximum(1. - y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon))))
    return -(one_loss + tf.constant(true_loss)) / tf.cast(tf.size(y_pred), tf.float32)

'''
def sparse_bce(y_true, y_pred, epsilon=1e-2):
    idx = y_true.indices
    false_mat = tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))).numpy()
    true_loss = 0
    for i in range(len(idx)):
        true_loss += tf.math.log(tf.math.maximum(y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon)))
        false_mat[idx[i][0], idx[i][1]] = 0
    return -(tf.math.reduce_mean(false_mat)/ tf.size(y_pred) + true_loss / tf.size(y_pred))
'''

'''
def sparse_bce(y_true, y_pred, epsilon=1e-4):
    idx = y_true.indices
    loss_mat = tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))).numpy()
    for i in range(len(idx)):
        loss_mat[idx[i][0], idx[i][1]] = \
            tf.math.log(tf.math.maximum(y_pred[idx[i][0], idx[i][1]], tf.constant(epsilon)))
    return -tf.math.reduce_mean(tf.convert_to_tensor(loss_mat))
'''
