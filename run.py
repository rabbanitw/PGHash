import tensorflow as tf
from dataloader import load_amazon670
from train import train
from network import Graph
from communicators import DecentralizedSGD, CentralizedSGD
from mlp import SparseNeuralNetwork
from unpack import get_model_architecture
from mpi4py import MPI


if __name__ == '__main__':

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # hashing parameters
    sdim = 8
    num_tables = 50
    cr = 0.1

    # load base network topology
    graph_type = 'ring'
    weight_type = None
    num_clusters = None
    G = Graph(rank, size, MPI.COMM_WORLD, graph_type, weight_type, num_c=num_clusters)

    # initialize model
    layer_dims = [135909, 128, 670091]

    initializer = tf.keras.initializers.GlorotUniform()
    initial_final_dense = initializer(shape=(128, 670091)).numpy()
    num_c_layers = int(cr*670091)

    worker_layer_dims = [135909, 128, num_c_layers]
    # worker_layer_dims = [135909, 128, 670091]

    model = SparseNeuralNetwork(worker_layer_dims)
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    batch_size = 64
    epochs = 2
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output,
    )

    # get model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    # initialize D-SGD or Centralized SGD
    # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
    communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1/size, layer_shapes, layer_sizes, 0, 1)

    print('Loading and partitioning data...')
    train_data, test_data = load_amazon670(rank, size, batch_size)

    print('Beginning training...')
    train(model, optimizer, communicator, train_data, test_data, epochs)
