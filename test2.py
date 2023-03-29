import numpy as np
import time
import tensorflow as tf
from itertools import product

if __name__ == '__main__':


    '''
    t4 = tf.constant([[0, 5],
                      [1, 6],
                      [2, 7],
                      [3, 8],
                      [4, 9]])

    update = tf.zeros([3, 2], dtype=tf.int32)

    print(t4.shape)

    print(tf.tensor_scatter_nd_update(t4, indices=[[2], [3], [0]], updates=update, name=None))

    print(tf.gather_nd(t4,
                       indices=[[2], [3], [0]]))

    t = tf.random.uniform(shape=[10])

    u = tf.zeros([3], dtype=tf.float32)

    print(tf.tensor_scatter_nd_update(t, indices=[[2], [3], [0]], updates=u, name=None))
    '''



    with tf.device('cpu'):
        A = tf.random.uniform(shape=(10000, 100000))
        x = tf.random.uniform(shape=(100000, 100))

        t = time.time()
        b = tf.matmul(A, x)
        print(time.time()-t)

        t = time.time()
        for i in range(10000):
            subA = tf.gather_nd(A, indices=[[i]])
        print(time.time()-t)

    '''
    x = [i for i in product(range(2), repeat=11)]
    x = np.array(x).T
    base2_hash = x.T.dot(1 << np.arange(x.T.shape[-1]))

    outer_dict = {}
    for hash_num in range(x.shape[1]):
        hash = x[:, hash_num, np.newaxis]
        hamm_dists = np.count_nonzero(x != hash, axis=0)
        inner_dict = {}
        for i in range(len(hamm_dists)):
            inner_dict[base2_hash[i]] = hamm_dists[i]
        outer_dict[base2_hash[hash_num]] = inner_dict

    print(outer_dict[3][2])
    '''



    '''
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
