import tensorflow as tf
import numpy as np
import argparse
from dataloader import load_extreme_data
from communicators import CentralizedSGD
from mpi4py import MPI
from misc import AverageMeter, Recorder, compute_accuracy_lsh
from unpack import PGHash
from mlp import SparseNeuralNetwork
import time
import resource
import os
import datetime
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def train(rank, PGHash, optimizer, communicator, train_data, test_data, num_labels, args):

    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, y_pred

    def get_partial_label(y_data, used_idx, batch_size, full_labels=num_labels):
        true_idx = y_data.indices.numpy()
        y_true = np.zeros((batch_size, full_labels))
        for i in true_idx:
            y_true[i[0], i[1]] = 1
        return tf.convert_to_tensor(y_true[:, used_idx], dtype=tf.float32)

    def get_memory(filename):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(filename, 'a') as f:
            # Dump timestamp, PID and amount of RAM.
            f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

    # hashing parameters
    lsh = args.lsh
    steps_per_lsh = args.steps_per_lsh
    cur_idx = None
    model = None

    # training parameters
    epochs = args.epochs

    top1 = AverageMeter()
    test_top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', args.name, MPI.COMM_WORLD.Get_size(), rank, hash_type)
    total_batches = 0
    test_acc = np.NaN

    fname = 'r{}.log'.format(rank)
    if os.path.exists(fname):
        os.remove(fname)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):

            if step == 0:
                # compute LSH
                cur_idx = PGHash.run_lsh(x_batch_train)
                # get new model
                model = PGHash.get_new_model()

            get_memory(fname)

            init_time = time.time()

            batch = x_batch_train.get_shape()[0]
            y_true = get_partial_label(y_batch_train, cur_idx, batch)
            loss_value, y_pred = train_step(x_batch_train, y_true)

            comm_time = 0
            lsh_time = 0
            # communication happens here
            # comm_time = communicator.communicate(model)

            # compute accuracy for the minibatch (top 1) & store accuracy and loss values
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)

            acc1 = compute_accuracy_lsh(y_pred, y_batch_train, cur_idx, num_labels)
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
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, [%d Total Samples]"
                    % (rank, step, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                )

            if step % 20 == 0:
                if rank == 0:
                    for (x_batch_test, y_batch_test) in test_data:
                        test_batch = x_batch_test.get_shape()[0]
                        y_pred_test = model(x_batch_test, training=False)
                        test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, cur_idx, num_labels)
                        test_top1.update(test_acc1, test_batch)
                    test_acc = test_top1.avg
                    '''
                    test_acc = PGHash.test_full_model(test_data, test_top1)
                    '''
                    print("Step %d: Top 1 Test Accuracy %.4f" % (step, test_acc))
                    recorder.add_testacc(test_acc)
                    test_top1.reset()

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
    return recorder.get_saveFolder()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # Add an argument
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Amazon670K')
    parser.add_argument('--graph_type', type=str, default='ring')
    parser.add_argument('--hash_type', type=str, default='slide_avg')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--lsh', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--sdim', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--cr', type=float, default=0.1)
    parser.add_argument('--train_bs', type=int, default=32)
    parser.add_argument('--test_bs', type=int, default=8192)
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
    train_bs = args.train_bs
    test_bs = args.test_bs
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = 'Data/' + args.dataset + '/train.txt'
    test_data_path = 'Data/' + args.dataset + '/test.txt'

    print('Loading and partitioning data...')
    train_data, test_data, n_features, n_labels = load_extreme_data(rank, size, train_bs, test_bs,
                                                                    train_data_path, test_data_path)

    print('Initializing model...')

    # initialize model
    PGHash = PGHash(n_labels, n_features, hls, sdim, num_tables, cr, hash_type)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    layer_shapes, layer_sizes = PGHash.get_model_architecture()

    # initialize D-SGD or Centralized SGD
    # communicator = DecentralizedSGD(rank, size, MPI.COMM_WORLD, G, layer_shapes, layer_sizes, 0, 1)
    communicator = CentralizedSGD(rank, size, MPI.COMM_WORLD, 1 / size, layer_shapes, layer_sizes, 0, 1)

    print('Beginning training...')
    saveFolder = train(rank, PGHash, optimizer, communicator, train_data, test_data, n_labels, args)

    # recv_indices = None
    # if rank == 0:
    #     recv_indices = np.empty_like(used_indices)
    # MPI.COMM_WORLD.Reduce(used_indices, recv_indices, op=MPI.SUM, root=0)

