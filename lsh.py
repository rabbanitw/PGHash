import numpy as np
import tensorflow as tf

#Takes a layer input and determines which weights are cosin (dis)similar via PGHash
#Parameters:
#in_layer = layer input
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
#RETURNS: row indices
def pglayer(in_layer,weight, sdim, num_tables, cr):
         
    #Instead of a normal random vector, it's going to be a repeating Gaussian vector.
    def pghash(vec1, vec2,n, sdim):
        sig1=np.zeros(sdim)
        sig2=np.zeros(sdim)
        for i in range(sdim):
            rand_vec=(1/np.sqrt(int(sdim)))*tf.random.normal([sdim])
            pg_vec=rand_vec
            for _ in range(int(n/sdim)-1):
                pg_vec=tf.concat([pg_vec,rand_vec],0)
            #If sdim does not evenly divide n:
            if int(n/sdim)*sdim != int(n):
                diff=n-int(n/sdim)*sdim
                pg_vec=np.concatenate([pg_vec,rand_vec[:diff]],0)
            sign1=np.sign(np.dot(pg_vec,vec1))
            sign2=np.sign(np.dot(pg_vec,vec2))
            if sign1<=0:
                sign1=0
            if sign2<=0:
                sign2=0
            sig1[i]=sign1
            sig2[i]=sign2
        return np.count_nonzero(sig1!=sig2)
    
    #Calculates PGHash similarities between a vector and a list.
    def pg_tables(v,vectors, n, sdim, num_tables):
        ham_avgs=[]
        #Loop over weights
        for i in range(vectors.shape[0]):
            hamming_dists=[]
            for _ in range(num_tables):
                hamming_dists.append(pghash(v,vectors[i], weight.shape[1], sdim))
            ham_avgs.append(np.average(hamming_dists))
        return np.asarray(ham_avgs)
    
    #Get the # of indices we'd like to keep.
    rows=weight.shape[0]
    num_inds = int(rows*cr)
    ham_avgs=pg_tables(in_layer, weight, weight.shape[1], sdim, num_tables)
    #Get vectors with high hamming distance (far angles)
    close_inds = (tf.math.top_k(ham_avgs,int(num_inds/2))).indices.numpy()
    #Get vectors with low hamming distance (close angles)
    far_inds= (tf.math.top_k(-1*ham_avgs,int(num_inds/2))).indices.numpy()
    inds = list(close_inds)+list(far_inds)

    return inds


#Takes a layer input and determines which weights are close/far in angle via SLIDE
#Parameters:
#in_layer = layer input
#weight = weight tensor of current layer
#sdim = length of hash signature. must divide length of in_layer/# of weight rows
#num_tables = how many hash tables to compare across
#cr = compression rate, percentage of rows of weight matrix to preserve
#RETURNS: row indices
def slidelayer(in_layer,weight, sdim, num_tables, cr):
         
    #Instead of a normal random vector, it's going to be a repeating Gaussian vector.
    def slidehash(vec1, vec2, n, sdim):
        sig1=np.zeros(sdim)
        sig2=np.zeros(sdim)
        for i in range(sdim):
            rand_vec=(1/np.sqrt(int(sdim)))*tf.random.normal([n])
            sign1=np.sign(np.dot(rand_vec,vec1))
            sign2=np.sign(np.dot(rand_vec,vec2))
            if sign1<=0:
                sign1=0
            if sign2<=0:
                sign2=0
            sig1[i]=sign1
            sig2[i]=sign2
        return np.count_nonzero(sig1!=sig2)
    
    #Calculates PGHash similarities between a vector and a list.
    def slide_tables(v,vectors, n, sdim, num_tables):
        ham_avgs=[]
        #Loop over weights
        for i in range(vectors.shape[0]):
            hamming_dists=[]
            for _ in range(num_tables):
                hamming_dists.append(slidehash(v,vectors[i], weight.shape[1], sdim))
            ham_avgs.append(np.average(hamming_dists))
        return np.asarray(ham_avgs)
    
    #Get the # of indices we'd like to keep.
    rows=weight.shape[0]
    num_inds = int(rows*cr)
    ham_avgs=slide_tables(in_layer, weight, weight.shape[1], sdim, num_tables)
    #Get vectors with high hamming distance (far angles)
    close_inds = (tf.math.top_k(ham_avgs,int(num_inds/2))).indices.numpy()
    #Get vectors with low hamming distance (close angles)
    far_inds= (tf.math.top_k(-1*ham_avgs,int(num_inds/2))).indices.numpy()
    inds = list(close_inds)+list(far_inds)

    #Resulting tensor is very sparse.
    return inds
