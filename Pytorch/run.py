import tensorflow as tf
import numpy as np
import argparse
from dataloader import load_extreme_data
from train import train
from mlp import NeuralNetwork, SparseNeuralNetwork
from mpi4py import MPI
import os

import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from torchsummary import summary
cudnn.benchmark = True
# from network import Graph
# from communicators import CentralizedSGD, LSHCentralizedSGD
# from unpack import get_model_architecture, flatten_tensors


def softmax_cross_entropy_with_logits(logits, labels, dim=-1):
    return -(labels * torch.nn.functional.log_softmax(logits, dim=dim)).sum(dim=dim)


class SoftmaxCELogits(torch.nn.Module):
    """
    Softmax Cross Entropy with Logits
    """

    def __init__(self):
        super(SoftmaxCELogits, self).__init__()

    def forward(self, logits, labels, dim=-1):
        losses = -torch.sum(labels * torch.nn.functional.log_softmax(logits, dim=dim))
        # losses = -torch.sum(torch.mul(labels, torch.nn.LogSoftmax()(logits)), dim=dim)
        return losses


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
    parser.add_argument('--epochs', type=int, default=10)
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


    torch.set_default_dtype(torch.float32)

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
    # train_data, tf_data, n_features, n_labels = load_extreme_data(rank, size, batch_size, train_data_path, test_data_path)
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
    # n_features = 10
    # n_labels = 5
    # hls = 10

    n_features = 782585
    n_labels = 205443

    torch_model = NeuralNetwork(n_features, hls, n_labels)
    learning_rate = args.lr/100
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(torch_model.parameters(), lr=args.lr)

    # initialize D-SGD or Centralized SGD
    # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
    # communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, 0, 1)
    # full_model = flatten_weights(model.get_weights())


    # print(model)

    MPI.COMM_WORLD.Barrier()
    # print('Beginning training...')
    #full_model, used_indices, saveFolder = train(rank, torch_model, optimizer, train_data, test_data, n_features,
    #                                          n_labels, args)


    # ======== TEST =========
    # '''
    X = np.random.rand(batch_size, n_features)
    Y = np.random.rand(batch_size, n_labels)

    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(Y)

    my_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)  # create your datset
    train_data = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size)  # create your dataloader

    tf_data = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size)
    

    worker_layer_dims = [n_features, hls, n_labels]

    model2 = SparseNeuralNetwork(worker_layer_dims)


    w1 = torch_model.layer1.weight.detach().numpy()
    b1 = torch_model.layer1.bias.detach().numpy()
    w2 = torch_model.layer2.weight.detach().numpy()
    b2 = torch_model.layer2.bias.detach().numpy()

    weights = model2.get_weights()
    weights[0] = w1.T
    weights[1] = b1
    weights[2] = w2.T
    weights[3] = b2
    model2.set_weights(weights)

    w1_torch = torch_model.layer1.weight.detach().numpy().T
    weights = model2.get_weights()
    w1_tf = weights[0]
    print('Diff in First Layer Weights (Start)')
    print(np.linalg.norm(w1_tf-w1_torch))

    X = torch.rand(32, n_features)*10

    X2 = tf.convert_to_tensor(X.numpy())

    pred_torch = torch_model(X).detach().numpy()

    pred_tf = model2(X2).numpy()


    print(np.linalg.norm(pred_tf-pred_torch)/np.linalg.norm(pred_tf))

    # for step, (x_batch_train, y_batch_train) in enumerate(train_data):
    (x_batch_train, y_batch_train) = next(iter(train_data))
    (x_batch_train2, y_batch_train2) = next(iter(tf_data))

    #print(y_batch_train)
    #y_batch_train = y_batch_train / torch.sum(y_batch_train)
    #print(y_batch_train)
    #### TRAIN COMPARISON

    # optimizer = torch.optim.Adam(torch_model.parameters(), lr=args.lr, eps=1e-7)
    optimizer = torch.optim.SGD(torch_model.parameters(), lr=args.lr)
    # criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    criterion = torch.nn.BCEWithLogitsLoss()
    # criterion = torch.nn.CrossEntropyLoss()
    # criterion = SoftmaxCELogits()
    torch_model.train(True)

    outputs = torch_model(x_batch_train)
    # outputs = torch.nn.Softmax()(outputs)
    # Compute the loss and its gradients
    # loss = torch.mean(softmax_cross_entropy_with_logits(outputs, y_batch_train))

    np_outputs = outputs.detach().numpy()
    np_label = y_batch_train.detach().numpy()
    tf_outputs = tf.convert_to_tensor(np_outputs)
    tf_label = tf.convert_to_tensor(np_label)
    tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_label, logits=tf_outputs))
    print(tf_loss)
    loss = torch.tensor(tf_loss.numpy(), requires_grad=True)


    # loss = criterion(outputs, y_batch_train)
    # loss = torch.nn.functional.cross_entropy(outputs, y_batch_train)
    loss.backward()
    # print(torch_model.layer1.weight.grad)
    # print(torch_model.layer1.bias.grad)
    optimizer.step()
    optimizer.zero_grad()

    print(loss)

    # optimizer2 = tf.keras.optimizers.Adam(learning_rate=args.lr)
    optimizer2 = tf.keras.optimizers.legacy.SGD(learning_rate=args.lr)
    # y_true = y_batch_train2 / tf.math.reduce_sum(y_batch_train2)
    y_true = y_batch_train2

    with tf.GradientTape() as tape:
        y_pred = model2(x_batch_train2, training=True)
        loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        # loss_value = lossF(y_batch_train2, y_pred)
    grads = tape.gradient(loss_value, model2.trainable_weights)
    # loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_batch_train2, logits=y_pred))

    
    #for var, g in zip(model2.trainable_variables, grads):
    #    # in this loop g is the gradient of each layer
    #    print(f'{var.name}, shape: {g.shape}')
    #    print("gradients..")
    #    print(tf.reduce_max(g))
    

    optimizer2.apply_gradients(zip(grads, model2.trainable_weights))

    print(loss_value)

    w1_torch = torch_model.layer1.weight.detach().numpy().T
    weights = model2.get_weights()
    w1_tf = weights[0]

    print('Diff in First Layer Weights (End)')
    print(np.linalg.norm(w1_tf - w1_torch))

    new_pred_torch = torch_model(X).detach().numpy()
    new_pred_tf = model2(X2).numpy()
    # print(np.linalg.norm(pred_torch - new_pred_torch))
    # print(np.linalg.norm(pred_tf - new_pred_tf))

    print(np.linalg.norm(new_pred_tf-new_pred_torch)/np.linalg.norm(new_pred_tf))

    # '''



