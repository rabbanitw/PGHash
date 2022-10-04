import tensorflow as tf


def get_custom_bce(epsilon=1e-2):
    def custom_bce(y_true, y_pred):
        return -tf.math.reduce_mean(y_true * tf.math.log(tf.math.maximum(y_pred, tf.constant(epsilon)))
                                    + (1. - y_true) * tf.math.log(tf.math.maximum(1. - y_pred, tf.constant(epsilon))))
    return custom_bce
