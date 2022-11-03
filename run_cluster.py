import tensorflow as tf
import numpy as np
import argparse
from dataloader import load_extreme_data
from train_cluster import train
from network import Graph
from communicators import DecentralizedSGD, CentralizedSGD, LSHCentralizedSGD
from mlp import SparseNeuralNetwork
from unpack import get_model_architecture, flatten_weights
from mpi4py import MPI
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--dataset', type=str, default='Amazon670k')
    parser.add_argument('--graph_type', type=str, default='ring')
    parser.add_argument('--hash_type', type=str, default='slide_vanilla')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--lsh', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--sdim', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--cr', type=int, default=0.1)
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

    # Assign GPUs and store CPU name
    cpu = tf.config.list_logical_devices('CPU')[0].name
    gpus = tf.config.list_logical_devices('GPU')
    gpu_names = []
    for gpu in gpus:
        gpu_names.append(gpu.name)
    num_gpus = len(gpu_names)
    gpu_id = rank % num_gpus
    gpu = gpu_names[gpu_id]

    # training parameters
    batch_size = args.batch_size
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = 'Data/' + args.dataset + '/train.txt'
    test_data_path = 'Data/' + args.dataset + '/test.txt'

    print('Loading and partitioning data...')
    with tf.device(cpu):
        train_data, test_data, n_features, n_labels = load_extreme_data(rank, size, batch_size,
                                                                        train_data_path, test_data_path)

    print('Initializing model...')
    # initialize full final dense layer (Glorot Uniform)
    sd = np.sqrt(6.0 / (hls + n_labels))
    initial_final_dense = np.random.uniform(-sd, sd, (hls, n_labels)).astype(np.float32)
    final_dense_shape = initial_final_dense.T.shape
    num_c_layers = int(cr * n_labels)

    if lsh:
        worker_layer_dims = [n_features, hls, num_c_layers]

    else:
        worker_layer_dims = [n_features, hls, n_labels]

    # Load compressed model onto designated GPU
    with tf.device(gpu):

        model = SparseNeuralNetwork(worker_layer_dims)

        full_model = None
        # get model architecture
        layer_shapes, layer_sizes = get_model_architecture(model)

        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    # load full model onto CPU and NOT the GPU (due to memory issues)
    with tf.device(cpu):
        if lsh:
            partial_model = flatten_weights(model.get_weights())
            partial_size = partial_model.size
            half_model_size = (n_features * hls) + hls + (4 * hls)
            missing_bias = n_labels-num_c_layers
            missing_weights = hls*(n_labels-num_c_layers)
            full_dense_weights = hls*n_labels
            full_model = np.zeros(partial_size + missing_weights + missing_bias)
            full_model[:half_model_size] = partial_model[:half_model_size]
            full_model[half_model_size:(half_model_size+full_dense_weights)] = initial_final_dense.T.flatten()
            # initialize Centralized LSH Communicator
            communicator = LSHCentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, layer_shapes, layer_sizes, 0, 1,
                                             n_features, n_labels, hls)
        else:
            # initialize D-SGD or Centralized SGD
            # communicator = DecentralizedSGD(rank, sieze, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
            communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, layer_shapes, layer_sizes, 0, 1)
            full_model = flatten_weights(model.get_weights())

    print('Beginning training...')
    full_model, used_indices, saveFolder = train(rank, model, optimizer, communicator, train_data, test_data,
                                                 full_model, epochs, gpu, cpu, sdim, num_tables,
                                                 n_features, n_labels, hls, cr, lsh, hash_type, steps_per_lsh)

    recv_indices = None
    if rank == 0:
        recv_indices = np.empty_like(used_indices)
    MPI.COMM_WORLD.Reduce(used_indices, recv_indices, op=MPI.SUM, root=0)

    if rank == 0:
        np.save(saveFolder + '/global_weight_frequency.npy', recv_indices)
        np.save(saveFolder + '/final_global_model.npy', full_model)
