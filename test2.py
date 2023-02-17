import numpy as np
import time
import tensorflow as tf

if __name__ == '__main__':
    # '''

    y_pred = np.random.rand(5, 3)
    print(y_pred)
    y_pred_t = tf.convert_to_tensor(y_pred.flatten())

    # indices = tf.constant([[1], [3], [4]])
    a = np.arange(3)
    ind = [[x, y] for x in range(5) for y in a]
    indices = tf.constant(ind)
    print(indices)
    shape = tf.constant([5, 5])
    scatter = tf.scatter_nd(indices, y_pred_t, shape)
    print(indices.shape)
    print(y_pred_t.shape)
    print(shape.shape)
    print(scatter)

    '''
    indices = tf.constant([[1], [3]])
    updates = tf.constant([[[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]],
                           [[5, 5, 5, 5], [6, 6, 6, 6],
                            [7, 7, 7, 7], [8, 8, 8, 8]]])
    print(updates.shape)
    updates = tf.convert_to_tensor(np.random.rand(2,4,4))
    shape = tf.constant([4, 4, 4])
    scatter = tf.scatter_nd(indices, updates, shape)
    print(scatter)
    '''
