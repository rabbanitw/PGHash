from typing import Union
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import (coo_matrix,
                          csr_matrix,
                          vstack)
from xclib.data import data_utils


class SparseDataset:
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, data: Union[np.ndarray, coo_matrix, csr_matrix],
                 targets: Union[np.ndarray, coo_matrix, csr_matrix],
                 transform: bool = None):

        # Transform data coo_matrix to csr_matrix for indexing
        if type(data) == coo_matrix:
            self.data = data.tocsr()
        else:
            self.data = data

        # Transform targets coo_matrix to csr_matrix for indexing
        if type(targets) == coo_matrix:
            self.targets = targets.tocsr()
        else:
            self.targets = targets

        self.transform = transform  # Can be removed

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def sparse_coo_to_tensor(coo: coo_matrix):
    """
    Transform scipy coo matrix to pytorch sparse tensor
    """
    values = coo.data
    indices = (coo.row, coo.col)  # np.vstack
    shape = coo.shape

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    s = torch.Size(shape)

    return torch.sparse.FloatTensor(i, v, s)


def sparse_batch_collate(batch):
    """
    Collate function which to transform scipy coo matrix to pytorch sparse tensor
    """
    # batch[0] since it is returned as a one element list
    data_batch, targets_batch = batch[0]

    if type(data_batch[0]) == csr_matrix:
        data_batch = data_batch.tocoo()  # removed vstack
        data_batch = sparse_coo_to_tensor(data_batch)
    else:
        data_batch = torch.FloatTensor(data_batch)

    if type(targets_batch[0]) == csr_matrix:
        targets_batch = targets_batch.tocoo()  # removed vstack
        targets_batch = sparse_coo_to_tensor(targets_batch)
    else:
        targets_batch = torch.FloatTensor(targets_batch)
    return data_batch, targets_batch



def load_extreme_data(rank, size, batch_size, train_data_path, test_data_path):
    # load data
    features, labels, num_samples, num_features, num_labels = data_utils.read_data(train_data_path)
    features_t, labels_t, num_labels_t, num_features_t, num_labels_t = data_utils.read_data(test_data_path)

    # size = 10

    # partition data amongst workers
    worker_features = partition_sparse_dataset(features, rank, size)
    worker_labels = partition_sparse_dataset(labels, rank, size)
    worker_features_t = partition_sparse_dataset(features_t, rank, size)
    worker_labels_t = partition_sparse_dataset(labels_t, rank, size)

    '''
    X = worker_features.todense()
    Y = worker_labels.todense()

    if rank == 0:
        print(X.shape)

    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(Y)

    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_dl = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size)  # create your dataloader

    trn_dataset = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    '''

    # '''
    # Create sparse tensors
    train_ds = SparseDataset(worker_features, worker_labels)
    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(train_ds), #, generator=torch.Generator(device='cuda')),
        batch_size=batch_size, drop_last=False)
    train_dl = DataLoader(train_ds,
                          batch_size=1,
                          collate_fn=sparse_batch_collate,
                          # generator=torch.Generator(device='cuda'),
                          sampler=sampler)
    # '''
    trn_dataset = None
    return train_dl, trn_dataset, num_features, num_labels


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


'''
    train_dl = DataLoader(train_ds,
                          batch_size=batch_size,
                          collate_fn=sparse_batch_collate,
                          shuffle=True)
    '''