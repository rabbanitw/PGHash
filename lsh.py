import numpy as np
import random
import tensorflow as tf

#Takes a layer input and determines which weights are cosin (dis)similar via PGHash
#Parameters:
#in_layer = layer input, must be a column-vector
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
def pg_vanilla(in_layer,weight, sdim, num_tables, cr, batch=1):
    
    #Parameters:
    #limit = total number of required neurons
    def pghash(vec1, vectors, n, sdim):
        match_indices=[]
        pg_mat=np.eye(sdim)
        #Build PG_Hash Matrix, rows look like repeating Gaussians
        for i in range(int(np.ceil(n/sdim))):
            pg_mat=np.concatenate((pg_mat,np.eye(sdim)),1)
        rand_gauss=np.random.normal(size=(sdim,sdim))
        pg_mat=(1/int(n/sdim))*rand_gauss@pg_mat[:,:n]
        #Apply PGHash to input vector.
        sig1=np.heaviside(pg_mat@vec1,0)
        for j in range(weight.shape[0]):
            vec2=weight[j]
            sig2=np.heaviside(pg_mat@vec2,0)
            #Matching hash signatures
            if np.array_equal(sig1,sig2):
                match_indices.append(j)
        return match_indices

    n=weight.shape[1]
    thresh=int(cr*weight.shape[0]/batch)
    inds=[]
    #Loop over the desired number of tables.
    for _ in range(num_tables):
        inds += pghash(in_layer, weight, n, sdim)
        inds=list(set(inds))
        if len(inds)>thresh:
            return random.sample(inds,k=thresh)
    return inds


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
    thresh=int(cr*weight.shape[0]/batch)
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
    def pghash(vec1, vectors, n, sdim):
        dists=np.zeros(vectors.shape[0])
        pg_mat=np.eye(sdim)
        #Build PG_Hash Matrix, rows look like repeating Gaussians
        for i in range(int(np.ceil(n/sdim))):
            pg_mat=np.concatenate((pg_mat,np.eye(sdim)),1)
        rand_gauss=np.random.normal(size=(sdim,sdim))
        pg_mat=(1/int(n/sdim))*rand_gauss@pg_mat[:,:n]
        #Apply PGHash to input vector.
        sig1=np.heaviside(pg_mat@vec1,0)
        for j in range(weight.shape[0]):
            vec2=weight[j]
            sig2=np.heaviside(pg_mat@vec2,0)
            #Hamming distances between signatures
            dists[j]=(np.count_nonzero(sig1!=sig2))
        return dists

    n=weight.shape[1]
    rows=weight.shape[0]
    ham_dists=np.zeros(rows)
    thresh=int(rows*cr)
    #Loop over the desired number of tables.
    for _ in range(num_tables):
        ham_dists += pghash(in_layer, weight, n, sdim)
    inds=(tf.math.top_k((-1/num_tables)*ham_dists,thresh)).indices.numpy()
    return inds


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
