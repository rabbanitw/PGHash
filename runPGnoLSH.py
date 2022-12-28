import tensorflow as tf
import numpy as np
import argparse
from dataloader import load_extreme_data
from communicators import CentralizedSGD
from mlp import SparseNeuralNetwork
from unpack import get_model_architecture, flatten_weights
from mpi4py import MPI
import os
from misc import AverageMeter, Recorder
from unpack import get_sub_model, get_full_dense, get_model_architecture, unflatten_weights
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def run_lsh(model, data, final_dense_w, sdim, num_tables, cr, hash_type):

    # get input layer for LSH
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output,
    )
    in_layer = feature_extractor(data).numpy()

    # run LSH to find the most important weights
    if hash_type == "pg_vanilla":
        return pg_vanilla(in_layer, final_dense_w, sdim, num_tables, cr)
    elif hash_type == "pg_avg":
        return pg_avg(in_layer, final_dense_w, sdim, num_tables, cr)
    elif hash_type == "slide_vanilla":
        return slide_vanilla(in_layer, final_dense_w, sdim, num_tables, cr)
    elif hash_type == "slide_avg":
        return slide_avg(in_layer, final_dense_w, sdim, num_tables, cr)


def train(rank, model, optimizer, communicator, train_data, test_data, full_model,
          num_f, num_l, args):

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, y_pred

    def test_step(x, y, cur_idx):
        y_pred = model(x, training=False)
        acc_metric.update_state(y_pred, y)

    def get_partial_label(y_data, used_idx, batch_size, full_labels):
        true_idx = y_data.indices.numpy()
        y_true = np.zeros((batch_size, full_labels))
        for i in true_idx:
            y_true[i[0], i[1]] = 1
        return tf.convert_to_tensor(y_true[:, used_idx], dtype=tf.float32)

    # hashing parameters
    sdim = args.sdim
    num_tables = args.num_tables
    cr = args.cr
    lsh = args.lsh
    hash_type = args.hash_type
    steps_per_lsh = args.steps_per_lsh

    # training parameters
    epochs = args.epochs
    hls = args.hidden_layer_size

    top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', args.name, MPI.COMM_WORLD.Get_size(), rank, hash_type)
    total_batches = 0
    start_idx_b = full_model.size - num_l
    start_idx_w = ((num_f * hls) + hls + (4 * hls))
    used_idx = np.zeros(num_l)
    cur_idx = np.arange(num_l)
    test_acc = np.NaN

    cur_idx = tf.convert_to_tensor(cur_idx)
    acc_metric = tf.keras.metrics.TopKCategoricalAccuracy(k=1)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            lsh_time = 0
            init_time = time.time()

            batch = x_batch_train.get_shape()[0]
            y_true = get_partial_label(y_batch_train, cur_idx, batch, num_l)
            loss_value, y_pred = train_step(x_batch_train, y_true)

            comm_time = 0
            lsh_time = 0
            # communication happens here
            # comm_time = communicator.communicate(model)

            # compute accuracy for the minibatch (top 1) & store accuracy and loss values
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)
            acc1 = acc_metric(y_pred, y_true)
            top1.update(acc1, batch)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - (lsh_time + record_time)

            recorder.add_new(comp_time+comm_time, comp_time, comm_time, lsh_time, acc1, test_acc, loss_value.numpy(),
                             top1.avg, losses.avg)

            # Save data to output folder
            recorder.save_to_file()

            total_batches += batch
            # Log every 200 batches.
            if step % 10 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Accuracy %.4f, [%d Total Samples]"
                    % (rank, step, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                )

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
    return full_model, used_idx, recorder.get_saveFolder()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Amazon670K')
    parser.add_argument('--graph_type', type=str, default='ring')
    parser.add_argument('--hash_type', type=str, default='slide_vanilla')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--lsh', action=argparse.BooleanOptionalAction, default=False)
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
    batch_size = args.batch_size
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = 'Data/' + args.dataset + '/train.txt'
    test_data_path = 'Data/' + args.dataset + '/test.txt'

    print('Loading and partitioning data...')
    train_data, test_data, n_features, n_labels = load_extreme_data(rank, size, batch_size,
                                                                    train_data_path, test_data_path)

    print('Initializing model...')

    # initialize model
    initializer = tf.keras.initializers.GlorotUniform()
    initial_final_dense = initializer(shape=(hls, n_labels)).numpy()
    final_dense_shape = initial_final_dense.T.shape
    num_c_layers = int(cr*n_labels)
    worker_layer_dims = [n_features, hls, num_c_layers]
    model = SparseNeuralNetwork(worker_layer_dims)

    # get model architecture
    layer_shapes, layer_sizes = get_model_architecture(model)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    # optimizer = tf.keras.optimizers.legacy.SGD(learning_rate=args.lr)

    # initialize D-SGD or Centralized SGD
    # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
    communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, layer_shapes, layer_sizes, 0, 1)
    full_model = flatten_weights(model.get_weights())

    print('Beginning training...')
    full_model, used_indices, saveFolder = train(rank, model, optimizer, communicator, train_data, test_data,
                                                 full_model, n_features, n_labels, args)

    # recv_indices = None
    # if rank == 0:
    #     recv_indices = np.empty_like(used_indices)
    # MPI.COMM_WORLD.Reduce(used_indices, recv_indices, op=MPI.SUM, root=0)

