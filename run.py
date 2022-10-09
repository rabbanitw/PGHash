import tensorflow as tf
from dataloader import load_amazon670
from train import train
from network import Graph
from communicators import DecentralizedSGD, CentralizedSGD
from mlp import SparseNeuralNetwork
from unpack import get_model_architecture, layer_compression
from mpi4py import MPI


if __name__ == '__main__':

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # load base network topology
    graph_type = 'ring'
    weight_type = None
    num_clusters = None
    G = Graph(rank, size, MPI.COMM_WORLD, graph_type, weight_type, num_c=num_clusters)

    # initialize model
    layer_dims = [135909, 128, 670091]
    model = SparseNeuralNetwork(layer_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    batch_size = 64
    epochs = 2

    # get model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    # initialize D-SGD or Centralized SGD
    # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
    communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1/size, layer_shapes, layer_sizes, 0, 1)

    print('Loading and partitioning data...')
    train_data, test_data = load_amazon670(rank, size, batch_size)

    # layer_compression(model, [1, 4])

    print('Beginning training...')
    train(model, optimizer, communicator, train_data, test_data, epochs)
