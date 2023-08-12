import torch
import argparse
from dataloader import data_generator_csr
from misc import AverageMeter, Recorder
from pghash import PGHash
from pg_train import pg_train
import numpy as np
import random
from network import SparseNN, SimpleNN

from xclib.data import data_utils
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix, coo_matrix


class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, data, targets, batch_size, coo=True):

        self.batch_size = batch_size
        self.global_idx = 0
        if coo:
            self.data = data.tocsr()
            self.targets = targets.tocsr()
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Delicious200K')
    parser.add_argument('--graph_type', type=str, default='fully_connected')
    parser.add_argument('--hash_type', type=str, default='pghash')
    parser.add_argument('--dwta', type=int, default=0)
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--c', type=int, default=8)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--cr', type=float, default=1)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=128)
    parser.add_argument('--steps_per_lsh', type=int, default=50)
    parser.add_argument('--steps_per_test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_layer_size', type=int, default=128)

    # parse the argument
    args = parser.parse_args()

    # set random seed
    randomSeed = args.randomSeed
    torch.manual_seed(randomSeed)
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    torch.backends.cudnn.benchmark = False

    # determine torch device available (default to GPU if available)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        dev = ["cuda:" + str(i) for i in range(num_gpus)]
    else:
        num_gpus = 0
        dev = ["cpu"]

    if num_gpus < 2:
        device = torch.device(dev[0])
        device2 = device
    else:
        device = torch.device(dev[0])
        device2 = torch.device(dev[1])

    # hashing parameters
    c = args.c
    k = args.k
    num_tables = args.num_tables
    hash_type = args.hash_type
    steps_per_lsh = args.steps_per_lsh

    # load base network topology
    graph_type = args.graph_type
    weight_type = None
    num_clusters = None

    # training parameters
    train_bs = args.train_bs
    test_bs = args.test_bs
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = '../data/' + args.dataset + '/train.txt'
    test_data_path = '../data/' + args.dataset + '/test.txt'

    nc = None
    nf = None
    if args.dataset == 'Amazon670K':
        nc = 670091
        nf = 135909
    elif args.dataset == 'Delicious200K':
        nc = 205443
        nf = 782585

    if k > c and args.dwta == 1:
        print('Error: Compression Size Smaller than Hash Length for PGHash-D')
        exit()

    if args.hash_type[:2] == 'pg':
        method = 'PGHash'
        cr = args.cr
        slide = False
    else:
        method = 'SLIDE'
        cr = 1
        slide = True

    # load (large) dataset
    print('Loading and partitioning data...')
    # train_dl = data_generator_csr(train_data_path, train_bs, nf, nc)
    # test_dl = data_generator_csr(test_data_path, test_bs, nf, nc)

    features, labels, num_samples, num_features, num_labels = data_utils.read_data(train_data_path)
    features_t, labels_t, num_samples_t, num_features_t, num_labels_t = data_utils.read_data(test_data_path)

    sparse_dataset = SparseDataset(features, labels, train_bs, coo=True)
    # train_dl = DataLoader(sparse_dataset, batch_size=train_bs, shuffle=True)

    sparse_dataset_t = SparseDataset(features_t, labels_t, test_bs, coo=True)
    # test_dl = DataLoader(sparse_dataset_t, batch_size=test_bs, shuffle=True)

    sampler = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(sparse_dataset),
        batch_size=train_bs,
        drop_last=False)

    sampler_t = torch.utils.data.sampler.BatchSampler(
        torch.utils.data.sampler.RandomSampler(sparse_dataset_t),
        batch_size=test_bs,
        drop_last=False)

    train_dl = DataLoader(sparse_dataset, sampler=sampler, collate_fn=sparse_batch_collate)
    test_dl = DataLoader(sparse_dataset_t, sampler=sampler_t, collate_fn=sparse_batch_collate)

    # exit()

    # initialize meters
    top1 = AverageMeter()
    test_top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('../output', 1, 0, args)

    # select method used and begin training once all devices are ready
    print('Initializing model...')
    Method = PGHash(nc, nf, 0, 1, 1, device, args, slide=slide)
    # model = SparseNN(nf, hls, nc)
    model = SimpleNN(nf, hls, nc)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=1e-7)
    pg_train(0, model, Method, device, optimizer, train_dl, test_dl, losses, top1, test_top1, recorder, args)
