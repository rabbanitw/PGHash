import numpy as np
import tensorflow as tf
from xclib.data import data_utils


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


def load_amazon670(batch_size):
    features, labels, num_samples, num_features, num_labels = data_utils.read_data('Data/Amazon670k/train.txt')
    features_t, labels_t, num_labels_t, num_features_t, num_labels_t = data_utils.read_data('Data/Amazon670k/test.txt')
    # Create sparse tensors
    trn = convert_sparse_matrix_to_sparse_tensor(features)
    tst = convert_sparse_matrix_to_sparse_tensor(features_t)
    trn_labels = convert_sparse_matrix_to_sparse_tensor(labels)
    tst_labels = convert_sparse_matrix_to_sparse_tensor(labels_t)
    # Create train and test datasets.
    trn_dataset = tf.data.Dataset.from_tensor_slices((trn, trn_labels)).batch(batch_size)
    tst_dataset = tf.data.Dataset.from_tensor_slices((tst, tst_labels)).batch(batch_size)
    return trn_dataset, tst_dataset
