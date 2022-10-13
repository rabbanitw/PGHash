import numpy as np
import tensorflow as tf
import sklearn.metrics as sn

#Takes a layer input and determines which weights are cosin (dis)similar via PGHash
#Parameters:
#in_layer = layer input, must be a column-vector
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
def pg_vanilla(in_layer,weight, sdim, num_tables, cr):

    #Parameters:
    #limit = total number of required neurons
    def pghash(in_layers, vectors, n, sdim):

        # create gaussian matrix
        pg_mat = (1/int(n/sdim))*np.tile(np.random.normal(size=(sdim, sdim)), int(np.ceil(n/sdim)))[:, :n]

        # Apply PGHash to input vector.
        sig1 = np.heaviside(pg_mat@in_layers.T, 0)

        # Apply PGHash to weights.
        sig2 = np.heaviside(pg_mat@vectors, 0)

        # convert to base 10
        input_vals = sig1.T.dot(1 << np.arange(sig1.T.shape[-1]))
        weight_vals = sig2.T.dot(1 << np.arange(sig2.T.shape[-1]))
        return np.concatenate([np.where(weight_vals == i)[0] for i in input_vals])

    n, cols = weight.shape
    thresh = int(cols * cr)
    inds = []
    # Loop over the desired number of tables.
    for _ in range(num_tables):
        inds.append(pghash(in_layer, weight, n, sdim))
        u_idx = np.unique(np.concatenate(inds))
        idx_len = len(np.unique(np.concatenate(inds)))
        if idx_len >= thresh:
            return np.unique(np.concatenate(inds))

    inds, frequency = np.unique(np.concatenate(inds), return_counts=True)

    if len(inds) > thresh:
        # choose the weights proportional to how frequently they pop-up
        p = frequency / np.sum(frequency)
        return np.random.choice(inds, thresh, p=p, replace=False)
    else:
        diff = thresh - len(inds)
        possible_idx = np.setdiff1d(np.arange(cols), inds)
        new = np.random.choice(possible_idx, diff, replace=False)
        return np.concatenate((inds, new))


#Takes a layer input and determines which weights are cosin (dis)similar via PGHash
#Parameters:
#in_layer = layer input, must be a column-vector
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
def slide_vanilla(in_layer,weight, sdim, num_tables, cr):

    def slidehash(in_layers, vectors, n, sdim):

        # create gaussian matrix
        slide_mat = np.random.normal(size=(sdim, n))

        # Apply Slide to input vector.
        sig1 = np.heaviside(slide_mat@in_layers.T, 0)

        # Apply Slide to weights.
        sig2 = np.heaviside(slide_mat@vectors, 0)

        # convert to base 10
        input_vals = sig1.T.dot(1 << np.arange(sig1.T.shape[-1]))
        weight_vals = sig2.T.dot(1 << np.arange(sig2.T.shape[-1]))

        return np.concatenate([np.where(weight_vals == i)[0] for i in input_vals])

    n, cols = weight.shape
    thresh = int(cols * cr)

    '''
    def gen_idx(num_union, in_layer, weight, n, sdim):
        idx_a = slidehash(in_layer, weight, n, sdim)
        idx_b = slidehash(in_layer, weight, n, sdim)
        for _ in range(num_union-1):
            idx_a = np.union1d(idx_a, slidehash(in_layer, weight, n, sdim))
            idx_b = np.union1d(idx_b, slidehash(in_layer, weight, n, sdim))
        return np.intersect1d(idx_a, idx_b)
    inds = gen_idx(3, in_layer, weight, n, sdim)
    for i in range(8):
        print(len(inds))
        inds = np.intersect1d(inds, gen_idx(3, in_layer, weight, n, sdim))
    '''

    inds = slidehash(in_layer, weight, n, sdim)
    # Loop over the desired number of tables.
    for i in range(num_tables-1):
        inds = np.intersect1d(inds, slidehash(in_layer, weight, n, sdim))
        if len(inds) <= thresh:
            return inds
    return inds
    


#Takes a layer input and determines which weights are cosine (dis)similar via PGHash
#Parameters:
#in_layer = layer input, must be a column-vector
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
def pg_avg(in_layer, weight, sdim, num_tables, cr):
    
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

    n, cols = weight.shape
    bs = in_layer.shape[0]
    thresh = int(cols*cr)
    ham_dists = np.zeros(cols)
    # Loop over the desired number of tables.
    for _ in range(num_tables):
        ham_dists += pghash(in_layer, weight, n, sdim)

    # pick just the largest differences
    avg_ham_dists = -ham_dists / (bs*num_tables)
    return (tf.math.top_k(avg_ham_dists, thresh)).indices.numpy()


#Takes a layer input and determines which weights are cosine (dis)similar via PGHash
#Parameters:
#in_layer = layer input, must be a column-vector
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
def slide_avg(in_layer, weight, sdim, num_tables, cr):

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
    return (tf.math.top_k(avg_ham_dists, thresh)).indices.numpy()
