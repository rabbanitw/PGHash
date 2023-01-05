import tensorflow as tf
import numpy as np
import argparse
from dataloader import load_extreme_data
from mpi4py import MPI
from misc import AverageMeter, Recorder
from pg_hash2 import PGHash
from pg_train import pg_train
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train(rank, PGHash, optimizer, train_data, test_data, num_labels, num_features, args):

    # initialize meters
    top1 = AverageMeter()
    test_top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', args.name, MPI.COMM_WORLD.Get_size(), rank, hash_type)

    # begin training
    pg_train(rank, PGHash, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args, num_labels,
             num_features)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Delicious200K')
    parser.add_argument('--graph_type', type=str, default='ring')
    parser.add_argument('--hash_type', type=str, default='pg_avg')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--lsh', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sdim', type=int, default=9)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--cr', type=float, default=0.1)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=2048)
    parser.add_argument('--steps_per_lsh', type=int, default=50)
    parser.add_argument('--steps_per_test', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden_layer_size', type=int, default=128)
    parser.add_argument('--q', type=int, default=10)

    # parse the argument
    args = parser.parse_args()

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # set random seed
    randomSeed = args.randomSeed
    tf.random.set_seed(randomSeed + rank)
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
    train_bs = args.train_bs
    test_bs = args.test_bs
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = 'Data/' + args.dataset + '/train.txt'
    test_data_path = 'Data/' + args.dataset + '/test.txt'

    if args.hash_type[:2] == 'pg':
        batch_size = train_bs * args.q
    else:
        batch_size = train_bs

    # load (large) dataset
    print('Loading and partitioning data...')
    train_data, test_data, n_features, n_labels = load_extreme_data(rank, size, batch_size, test_bs,
                                                                    train_data_path, test_data_path)

    # initialize model
    print('Initializing model...')
    PGHash = PGHash(n_labels, n_features, hls, sdim, num_tables, cr, hash_type, rank, size, args.q, 1 / size, 0, 1)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    layer_shapes, layer_sizes = PGHash.get_model_architecture()

    # begin training
    print('Beginning training...')
    train(rank, PGHash, optimizer, train_data, test_data, n_labels, n_features, args)
