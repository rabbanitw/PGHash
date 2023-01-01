import numpy as np
import tensorflow as tf
from xclib.data import data_utils


def convert_sparse_matrix_to_sparse_tensor(X):
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    return tf.sparse.reorder(tf.SparseTensor(indices, coo.data, coo.shape))


def load_extreme_data(rank, size, train_bs, test_bs, train_data_path, test_data_path):
    # load data
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(train_data_path)
    features_t, labels_t, num_labels_t, num_features_t, num_labels_t = data_utils.read_data(test_data_path)

    # partition data amongst workers
    worker_features = partition_sparse_dataset(features, rank, size)
    worker_features_t = partition_sparse_dataset(features_t, rank, size)
    worker_labels = partition_sparse_dataset(labels, rank, size)
    worker_labels_t = partition_sparse_dataset(labels_t, rank, size)

    # Create sparse tensors
    trn = convert_sparse_matrix_to_sparse_tensor(worker_features)
    tst = convert_sparse_matrix_to_sparse_tensor(worker_features_t)
    trn_labels = convert_sparse_matrix_to_sparse_tensor(worker_labels)
    tst_labels = convert_sparse_matrix_to_sparse_tensor(worker_labels_t)

    # Create train and test datasets
    trn_dataset = tf.data.Dataset.from_tensor_slices((trn, trn_labels)).batch(train_bs)
    tst_dataset = tf.data.Dataset.from_tensor_slices((tst, tst_labels)).batch(test_bs)
    return trn_dataset, tst_dataset, num_features, num_labels


def partition_sparse_dataset(data_array, rank, size, partition_type='Simple'):
    if partition_type == 'Simple':
        r, c = data_array.shape
        start_idx = rank*int(r/size)
        end_idx = (rank+1)*int(r/size)
        if rank == size - 1:
            end_idx = r
        worker_data = data_array[start_idx:end_idx, :]
        return worker_data

    # To Do: implement other partition methods below...
    else:
        return np.array_split(data_array, size)[rank]
