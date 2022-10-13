import tensorflow as tf
import numpy as np
from dataloader import load_amazon670
from train import train
from network import Graph
from communicators import CentralizedSGD, LSHCentralizedSGD
from mlp import SparseNeuralNetwork
from unpack import get_model_architecture, flatten_weights
from mpi4py import MPI


if __name__ == '__main__':

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    randomSeed = 1203

    # set random seed
    tf.random.set_seed(randomSeed + rank)
    np.random.seed(randomSeed)

    # hashing parameters
    sdim = 8
    num_tables = 50
    cr = 0.1
    lsh = True

    # load base network topology
    graph_type = 'ring'
    weight_type = None
    num_clusters = None
    G = Graph(rank, size, MPI.COMM_WORLD, graph_type, weight_type, num_c=num_clusters)

    # initialize model
    initializer = tf.keras.initializers.GlorotUniform()
    initial_final_dense = initializer(shape=(128, 670091)).numpy()
    final_dense_shape = initial_final_dense.T.shape
    num_c_layers = int(cr*670091)

    if lsh:
        worker_layer_dims = [135909, 128, num_c_layers]

    else:
        worker_layer_dims = [135909, 128, 670091]

    model = SparseNeuralNetwork(worker_layer_dims)

    full_model = None
    # get model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    batch_size = 256
    epochs = 10

    if lsh:
        partial_model = flatten_weights(model.get_weights())
        partial_size = partial_model.size
        half_model_size = (135909 * 128) + 128 + (4 * 128)
        missing_bias = 670091-num_c_layers
        missing_weights = 128*(670091-num_c_layers)
        full_dense_weights = 128*670091
        full_model = np.zeros(partial_size + missing_weights + missing_bias)
        full_model[:half_model_size] = partial_model[:half_model_size]
        full_model[half_model_size:(half_model_size+full_dense_weights)] = initial_final_dense.T.flatten()
        # initialize Centralized LSH Communicator
        communicator = LSHCentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, layer_shapes, layer_sizes, 0, 1)
    else:
        # initialize D-SGD or Centralized SGD
        # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
        communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, layer_shapes, layer_sizes, 0, 1)

    print('Loading and partitioning data...')
    train_data, test_data = load_amazon670(rank, size, batch_size)

    print('Beginning training...')
    full_model, used_indices, saveFolder = train(rank, model, optimizer, communicator, train_data, test_data, full_model, epochs)

    recv_indices = None
    if rank == 0:
        recv_indices = np.empty_like(used_indices)
    MPI.COMM_WORLD.Reduce(used_indices, recv_indices, op=MPI.SUM, root=0)

    if rank == 0:
        np.save(saveFolder+'/global_weight_frequency.npy', recv_indices)
        np.save(saveFolder + '/final_global_model.npy', full_model)
