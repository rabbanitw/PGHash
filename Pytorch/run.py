# import tensorflow as tf
import numpy as np
import argparse
from dataloader import load_extreme_data
from train import train
# from network import Graph
# from communicators import CentralizedSGD, LSHCentralizedSGD
from mlp import NeuralNetwork
# from unpack import get_model_architecture, flatten_tensors
from mpi4py import MPI
import os

import torch
import torch.utils.data.distributed
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Amazon670k')
    parser.add_argument('--graph_type', type=str, default='ring')
    parser.add_argument('--hash_type', type=str, default='slide_vanilla')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--lsh', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sdim', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--cr', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--steps_per_lsh', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden_layer_size', type=int, default=128)

    # Parse the argument
    args = parser.parse_args()

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # set random seed
    randomSeed = args.randomSeed
    torch.manual_seed(args.randomSeed + rank)
    np.random.seed(randomSeed)

    # hashing parameters
    sdim = args.sdim
    num_tables = args.num_tables
    cr = args.cr
    lsh = args.lsh
    hash_type = args.hash_type
    steps_per_lsh = args.steps_per_lsh

    # load base network topology
    graph_type = args.graph_type
    weight_type = None
    num_clusters = None
    # G = Graph(rank, size, MPI.COMM_WORLD, graph_type, weight_type, num_c=num_clusters)

    # training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = '../Data/' + args.dataset + '/train.txt'
    test_data_path = '../Data/' + args.dataset + '/test.txt'

    print('Loading and partitioning data...')
    train_data, n_features, n_labels = load_extreme_data(rank, size, batch_size, train_data_path, test_data_path)
    test_data = None

    print('Initializing model...')

    '''
    # initialize model
    initializer = tf.keras.initializers.GlorotUniform()
    initial_final_dense = initializer(shape=(hls, n_labels)).numpy()
    final_dense_shape = initial_final_dense.T.shape
    num_c_layers = int(cr*n_labels)

    lsh = False

    if lsh:
        worker_layer_dims = [n_features, hls, num_c_layers]
    else:
        worker_layer_dims = [n_features, hls, n_labels]
    '''

    model = NeuralNetwork(n_features, hls, n_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # initialize D-SGD or Centralized SGD
    # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
    # communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, 0, 1)
    # full_model = flatten_weights(model.get_weights())

    print('Beginning training...')
    full_model, used_indices, saveFolder = train(rank, model, optimizer, train_data, test_data, n_features,
                                                 n_labels, args)


