import tensorflow as tf
import argparse
from util.dataloader import load_extreme_data
from mpi4py import MPI
from util.misc import AverageMeter, Recorder
from models.pghash import PGHash
from models.slide import SLIDE
from models.dense import ModelHub
from train.pg_train import pg_train
from train.regular_train import regular_train
from train.slide_train import slide_train
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Delicious200K')
    parser.add_argument('--graph_type', type=str, default='fully_connected')
    parser.add_argument('--hash_type', type=str, default='pghash')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--sdim', type=int, default=9)
    parser.add_argument('--num_tables', type=int, default=1)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--cr', type=float, default=0.1)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=2048)
    parser.add_argument('--steps_per_lsh', type=int, default=50)
    parser.add_argument('--steps_per_test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--hidden_layer_size', type=int, default=128)
    parser.add_argument('--q', type=int, default=50)

    # parse the argument
    args = parser.parse_args()

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # set random seed
    randomSeed = args.randomSeed
    tf.keras.utils.set_random_seed(randomSeed)

    # hashing parameters
    sdim = args.sdim
    num_tables = args.num_tables
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
    train_data_path = 'data/' + args.dataset + '/train.txt'
    test_data_path = 'data/' + args.dataset + '/test.txt'

    if args.hash_type[:2] == 'pg':
        batch_size = train_bs * args.q
        method = 'PGHash'
        cr = args.cr
    elif args.hash_type[:3] == 'reg':
        batch_size = train_bs
        method = 'Regular'
        cr = 1
    else:
        batch_size = train_bs
        method = 'SLIDE'
        cr = 1

    with tf.device('/CPU:0'):
        # load (large) dataset
        print('Loading and partitioning data...')
        train_data, test_data, n_features, n_labels = load_extreme_data(rank, size, batch_size, test_bs,
                                                                        train_data_path, test_data_path)

        # initialize meters
        top1 = AverageMeter()
        test_top1 = AverageMeter()
        losses = AverageMeter()
        recorder = Recorder('Output', MPI.COMM_WORLD.Get_size(), rank, args)

        # initialize model
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
        print('Initializing model...')
        if method == 'PGHash':
            Method = PGHash(n_labels, n_features, rank, size, 1 / size, args)
            # begin training once all devices are ready
            MPI.COMM_WORLD.Barrier()
            pg_train(rank, size, PGHash, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args)

        elif method == 'SLIDE':
            Method = SLIDE(n_labels, n_features, rank, size, 1 / size, args)
            # begin training once all devices are ready
            MPI.COMM_WORLD.Barrier()
            slide_train(rank, PGHash, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args)

        elif method == 'Regular':
            Method = ModelHub(n_labels, n_features, hls, sdim, num_tables, cr, hash_type, rank, size, args.q, 1 / size)
            # begin training once all devices are ready
            MPI.COMM_WORLD.Barrier()
            regular_train(rank, size, PGHash, optimizer, train_data, test_data, losses, top1, recorder, args)

        else:
            print('ERROR: No Method Selected')
            exit()
