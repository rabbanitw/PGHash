import numpy as np
from xclib.data import data_utils



def load_extreme_data(rank, size, train_bs, test_bs, train_data_path, test_data_path):
    # load data
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(train_data_path)
    features_t, labels_t, num_labels_t, num_features_t, num_labels_t = data_utils.read_data(test_data_path)

    # partition data amongst workers
    worker_features = partition_sparse_dataset(features, rank, size)
    worker_features_t = partition_sparse_dataset(features_t, rank, size)
    worker_labels = partition_sparse_dataset(labels, rank, size)
    worker_labels_t = partition_sparse_dataset(labels_t, rank, size)

    return (worker_features, worker_labels), (worker_features_t, worker_labels_t), num_features, num_labels


def partition_sparse_dataset(data_array, rank, size, partition_type='Simple'):
    if partition_type == 'Simple':
        r, c = data_array.shape
        val, rem = divmod(r, size)
        if rank < rem:
            start_idx = rank*val + rank
            end_idx = (rank+1)*val + rank + 1
        else:
            start_idx = rank * val + rem
            end_idx = (rank + 1) * val + rem
        worker_data = data_array[start_idx:end_idx, :]
        return worker_data

    # To Do: implement other partition methods below...
    else:
        return np.array_split(data_array, size)[rank]
