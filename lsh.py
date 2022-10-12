import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import sklearn.metrics as sn
import time

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
    for _ in tqdm(range(num_tables)):
        inds.append(pghash(in_layer, weight, n, sdim))
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
    
    #Parameters:
    #limit = total number of required neurons
    def slidehash(vec1, vectors, n, sdim):
        match_indices=[]
        slide_mat=np.random.normal(size=(sdim,n))
        #Apply PGHash to input vector.
        sig1=np.heaviside(slide_mat@vec1,0)
        for j in range(weight.shape[0]):
            vec2=weight[j]
            sig2=np.heaviside(slide_mat@vec2,0)
            #Matching hash signatures
            if np.array_equal(sig1,sig2):
                match_indices.append(j)
        return match_indices

    n=weight.shape[1]
    thresh=int(cr*weight.shape[0])
    inds=[]
    #Loop over the desired number of tables.
    for _ in range(num_tables):
        inds += slidehash(in_layer, weight, n, sdim)
        inds = list(set(inds))
        if len(inds)>thresh:
            return random.sample(inds,k=thresh)
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

        t = time.time()
        # create gaussian matrix
        pg_mat = (1/int(n/sdim))*np.tile(np.random.normal(size=(sdim, sdim)), int(np.ceil(n/sdim)))[:, :n]

        # Apply PGHash to input vector.
        sig1 = np.heaviside(pg_mat@in_layers.T, 0)

        # Apply PGHash to weights.
        sig2 = np.heaviside(pg_mat@vectors, 0)

        # Compute Hamming Distance
        v = sn.DistanceMetric.get_metric("hamming").pairwise(sig2.T, sig1.T) * sig2.T.shape[-1]
        # print(time.time()-t)

        return np.sum(v.T, axis=0)

    n, cols = weight.shape
    bs = in_layer.shape[0]
    thresh = int(cols*cr)
    ham_dists = np.zeros(cols)
    # Loop over the desired number of tables.
    for _ in tqdm(range(num_tables)):
        ham_dists += pghash(in_layer, weight, n, sdim)

    # pick just the largest differences
    avg_ham_dists = -ham_dists / (bs*num_tables)
    '''
    # Use for largest AND smallest differences
    avg_ham_dists = np.abs(ham_dists - (sdim / 2) * num_tables)
    avg_ham_dists = np.sum(avg_ham_dists, axis=0)/bs
    '''
    return (tf.math.top_k(avg_ham_dists, thresh)).indices.numpy()


#Takes a layer input and determines which weights are cosine (dis)similar via PGHash
#Parameters:
#in_layer = layer input, must be a column-vector
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
def slide_avg(in_layer,weight, sdim, num_tables, cr):
    
    #Parameters:
    #limit = total number of required neurons
    def slidehash(vec1, vectors, n, sdim):
        dists=np.zeros(vectors.shape[0])
        #Build random Gaussian matrix
        slide_mat=np.random.normal(size=(sdim, n))
        #Apply PGHash to input vector.
        sig1=np.heaviside(slide_mat@vec1,0)
        for j in range(weight.shape[0]):
            vec2=weight[j]
            sig2=np.heaviside(slide_mat@vec2,0)
            #Hamming distances between signatures
            dists[j]=(np.count_nonzero(sig1!=sig2))
        return dists

    n=weight.shape[1]
    rows=weight.shape[0]
    ham_dists=np.zeros(rows)
    thresh=int(rows*cr)
    #Loop over the desired number of tables.
    for _ in range(num_tables):
        ham_dists += slidehash(in_layer, weight, n, sdim)
    inds=(tf.math.top_k((-1/num_tables)*ham_dists,thresh)).indices.numpy()
    return inds
