import numpy as np
import tensorflow as tf
import sklearn.metrics as sn


'''
# Apply PGHash to input vector.
sig1 = np.heaviside(pg_mat@in_layers.T, 0)

# convert to base 10
input_vals = sig1.T.dot(1 << np.arange(sig1.T.shape[-1]))

# convert to base 10
weight_vals = sig2.T.dot(1 << np.arange(sig2.T.shape[-1]))

#Parameters:
#limit = total number of required neurons
def pghash(in_layers, vectors, n, sdim):

    # create gaussian matrix
    pg_mat = (1/int(n/sdim))*np.tile(np.random.normal(size=(sdim, sdim)), int(np.ceil(n/sdim)))[:, :n]

    # Apply PGHash to input vector.
    sig1 = np.heaviside(pg_mat@in_layers.T, 0)

    # Apply PGHash to weights.
    sig2 = np.heaviside(pg_mat@vectors, 0)

    # Compute Hamming Distance
    v = sn.DistanceMetric.get_metric("hamming").pairwise(sig2.T, sig1.T) * sig2.T.shape[-1]

    return np.sum(v.T, axis=0)
'''


#Parameters:
#limit = total number of required neurons
def pghash(weights, n, sdim):
    '''
    compute hashing
    :param vectors:
    :param n:
    :param sdim:
    :return:
    '''

    # create gaussian matrix
    pg_gaussian = (1/int(n/sdim))*np.tile(np.random.normal(size=(sdim, sdim)), int(np.ceil(n/sdim)))[:, :n]

    # Apply PGHash to weights.
    hash_table = np.heaviside(pg_gaussian@weights, 0)

    return pg_gaussian, hash_table


def slidehash(vectors, n, sdim):

    # create gaussian matrix
    slide_gaussian = np.random.normal(size=(sdim, n))

    # Apply Slide to weights.
    hash_table = np.heaviside(slide_gaussian@vectors, 0)

    # convert to base 10
    hash_table = hash_table.T.dot(1 << np.arange(hash_table.T.shape[-1]))

    return slide_gaussian, hash_table


def pg_vanilla(in_layer, weight, sdim, num_tables, cr):
    '''
    Takes a layer input and determines which weights are cosin (dis)similar via PGHash
    :param in_layer: layer input, must be a column-vector
    :param weight: weight tensor of current layer
    :param sdim: length of hash signature. must divide length of in_layer/# of weight rows
    :param num_tables: how many hash tables to compare across
    :param cr: compression rate, percentage of rows of weight matrix to preserve
    :return: hash tables and ...
    '''

    n, cols = weight.shape
    thresh = int(cols * cr)

    inds = pghash(in_layer, weight, n, sdim)
    # Loop over the desired number of tables.
    for i in range(num_tables - 1):
        if len(inds) <= thresh:
            return inds
        inds = np.intersect1d(inds, pghash(in_layer, weight, n, sdim))
    return np.sort(inds)


def pg_avg(in_layer, pg_gaussian, hash_table):
    '''
        Takes a layer input and determines which weights are cosin (dis)similar via PGHash
        :param in_layer: layer input, must be a column-vector
        :param pg_gaussian:
        :param hash_table:
        :return:
    '''

    # Apply PGHash to input data.
    data_hash = np.heaviside(pg_gaussian @ in_layer.T, 0)

    # Compute Hamming Distance
    v = sn.DistanceMetric.get_metric("hamming").pairwise(hash_table.T, data_hash.T) * hash_table.T.shape[-1]

    return np.sum(v.T, axis=0)


def slide_vanilla(in_layer, gaussian, weight_ht):
    '''
        Takes a layer input and determines which weights are cosin (dis)similar via PGHash
        :param in_layer: layer input, must be a column-vector
        :param weight: weight tensor of current layer
        :param sdim: length of hash signature. must divide length of in_layer/# of weight rows
        :param num_tables: how many hash tables to compare across
        :param cr: compression rate, percentage of rows of weight matrix to preserve
        :return: hash tables and ...
    '''

    # Apply Slide to input vector.
    transformed_layer = np.heaviside(gaussian @ in_layer.T, 0)
    input_vals = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))

    '''
        n, cols = weight.shape
        thresh = int(cols * cr)

        inds = slidehash(in_layer, weight, n, sdim)
        # Loop over the desired number of tables.
        for i in range(num_tables-1):
            if len(inds) <= thresh:
                return inds
            inds = np.intersect1d(inds, slidehash(in_layer, weight, n, sdim))
        return np.sort(inds)
        '''

    # map equivalent hashes and return them
    return np.concatenate([np.where(weight_ht == i)[0] for i in input_vals])


def slide_avg(in_layer, weight, sdim, num_tables, cr):
    '''
        Takes a layer input and determines which weights are cosin (dis)similar via PGHash
        :param in_layer: layer input, must be a column-vector
        :param weight: weight tensor of current layer
        :param sdim: length of hash signature. must divide length of in_layer/# of weight rows
        :param num_tables: how many hash tables to compare across
        :param cr: compression rate, percentage of rows of weight matrix to preserve
        :return: hash tables and ...
    '''

    def slidehash(in_layers, vectors, n, sdim):
        # create gaussian matrix
        slide_mat = np.random.normal(size=(sdim, n))

        # Apply PGHash to input vector.
        sig1 = np.heaviside(slide_mat @ in_layers.T, 0)

        # Apply PGHash to weights.
        sig2 = np.heaviside(slide_mat @ vectors, 0)

        # Compute Hamming Distance
        v = sn.DistanceMetric.get_metric("hamming").pairwise(sig2.T, sig1.T) * sig2.T.shape[-1]

        return np.sum(v.T, axis=0)

    n, cols = weight.shape
    bs = in_layer.shape[0]
    thresh = int(cols * cr)
    ham_dists = np.zeros(cols)
    # Loop over the desired number of tables.
    for _ in range(num_tables):
        ham_dists += slidehash(in_layer, weight, n, sdim)

    # pick just the largest differences
    avg_ham_dists = -ham_dists / (bs * num_tables)
    return np.sort((tf.math.top_k(avg_ham_dists, thresh)).indices.numpy())
