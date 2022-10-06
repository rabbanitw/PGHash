import tensorflow as tf
from dataloader import load_amazon670
from train import train
from network import Graph
from dsgd import DecentralizedSGD
from mlp import SparseNeuralNetwork
from unpack import get_model_architecture
from mpi4py import MPI


if __name__ == '__main__':

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # load base network topology
    graph_type = 'ring'
    weight_type = None
    num_clusters = None
    p = 3 / size
    G = Graph(rank, size, MPI.COMM_WORLD, graph_type, weight_type, p=p, num_c=num_clusters)

    # initialize model
    layer_dims = [135909, 128, 670091]
    model = SparseNeuralNetwork(layer_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    batch_size = 64
    epochs = 2

    # get model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    # initialize D-SGD
    communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)

    print('Loading and partitioning data...')
    train_data, test_data = load_amazon670(rank, size, batch_size)

    print('Beginning training...')
    train(model, optimizer, train_data, test_data, epochs)
