import tensorflow as tf
import argparse
from util.dataloader import load_extreme_data
from mpi4py import MPI
from util.misc import AverageMeter, Recorder
from models.pghash import PGHash
from models.slide import SLIDE
from models.dense import Dense
from train.pg_train import pg_train
from train.regular_train import regular_train
from train.slide_train import slide_train
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


class Adam(tf.Module):

    def __init__(self, learning_rate=1e-4, beta_1=0.9, beta_2=0.999, ep=1e-8):
        # Initialize the Adam parameters
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.learning_rate = learning_rate
        self.ep = ep
        self.t = 1.
        self.v_dvar, self.s_dvar = [], []
        self.title = f"Adam: learning rate={self.learning_rate}"
        self.built = False

    def apply_gradients(self, grads, vars, final_layer_active_neurons, final_layer_inactive_neurons, fli=2, fbi=3,
                        hls=128):
        # Set up moment and RMSprop slots for each variable on the first call
        if not self.built:
            for var in vars:
              v = tf.Variable(tf.zeros(shape=var.shape))
              s = tf.Variable(tf.zeros(shape=var.shape))
              self.v_dvar.append(v)
              self.s_dvar.append(s)
            self.built = True
        # Perform Adam updates
        active_idx = [[x] for x in final_layer_active_neurons]
        inactive_idx = [[x] for x in final_layer_inactive_neurons]
        if len(inactive_idx) == 0:
            flag = True
        else:
            flag = False
        zero_update_layer = tf.zeros([len(final_layer_inactive_neurons), hls], dtype=tf.float32)
        zero_update_bias = tf.zeros([len(final_layer_inactive_neurons)], dtype=tf.float32)
        for i, (d_var, var) in enumerate(zip(grads, vars)):
            if i < fli or flag:
                d_var = tf.convert_to_tensor(d_var)
                # Moment calculation
                self.v_dvar[i] = self.beta_1*self.v_dvar[i] + (1-self.beta_1)*d_var
                # RMSprop calculation
                self.s_dvar[i] = self.beta_2*self.s_dvar[i] + (1-self.beta_2)*tf.square(d_var)
                # Bias correction
                v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
                s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
            elif i == fli:
                # Isolate active indices
                sub_dvar = tf.transpose(tf.gather_nd(tf.transpose(d_var), indices=active_idx))
                # Moment update
                active_v_dvar = tf.transpose(tf.gather_nd(tf.transpose(self.v_dvar[i]), indices=active_idx))
                update = tf.transpose(self.beta_1 * active_v_dvar + (1 - self.beta_1) * sub_dvar)
                tf.tensor_scatter_nd_update(tf.transpose(self.v_dvar[i]), indices=active_idx, updates=update)
                # RMS update
                active_s_dvar = tf.transpose(tf.gather_nd(tf.transpose(self.s_dvar[i]), indices=active_idx))
                update = tf.transpose(self.beta_2 * active_s_dvar + (1 - self.beta_2) * tf.square(sub_dvar))
                tf.tensor_scatter_nd_update(tf.transpose(self.s_dvar[i]), indices=active_idx, updates=update)
                # Bias correction
                v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
                s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
                # set update portion to 0 for inactive neurons
                tf.tensor_scatter_nd_update(tf.transpose(v_dvar_bc), indices=inactive_idx, updates=zero_update_layer)
            elif i == fbi:
                # Isolate active indices
                sub_dvar = tf.gather_nd(d_var, indices=active_idx)
                # Moment update
                active_v_dvar = tf.gather_nd(self.v_dvar[i], indices=active_idx)
                update = self.beta_1 * active_v_dvar + (1 - self.beta_1) * sub_dvar
                tf.tensor_scatter_nd_update(self.v_dvar[i], indices=active_idx, updates=update)
                # RMS update
                active_s_dvar = tf.gather_nd(tf.transpose(self.s_dvar[i]), indices=active_idx)
                update = self.beta_2 * active_s_dvar + (1 - self.beta_2) * tf.square(sub_dvar)
                tf.tensor_scatter_nd_update(self.s_dvar[i], indices=active_idx, updates=update)
                # Bias correction
                v_dvar_bc = self.v_dvar[i] / (1 - (self.beta_1 ** self.t))
                s_dvar_bc = self.s_dvar[i] / (1 - (self.beta_2 ** self.t))
                # set update portion to 0 for inactive neurons
                tf.tensor_scatter_nd_update(v_dvar_bc, indices=inactive_idx, updates=zero_update_bias)

            # Update model variables
            var.assign_sub(self.learning_rate*(v_dvar_bc/(tf.sqrt(s_dvar_bc) + self.ep)))

        # Increment the iteration counter
        self.t += 1.


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Delicious200K')
    parser.add_argument('--graph_type', type=str, default='fully_connected')
    parser.add_argument('--hash_type', type=str, default='pghash')
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--sdim', type=int, default=8)
    parser.add_argument('--c', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--cr', type=float, default=0.2)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=4096)
    parser.add_argument('--steps_per_lsh', type=int, default=1)
    parser.add_argument('--steps_per_test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_layer_size', type=int, default=128)

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
        method = 'PGHash'
        cr = args.cr
    elif args.hash_type[:3] == 'reg':
        method = 'Regular'
        cr = 1
    else:
        method = 'SLIDE'
        cr = 1

    with tf.device('/CPU:0'):
        # load (large) dataset
        print('Loading and partitioning data...')
        train_data, test_data, n_features, n_labels = load_extreme_data(rank, size, train_bs, test_bs, train_data_path,
                                                                        test_data_path)

        # initialize meters
        top1 = AverageMeter()
        test_top1 = AverageMeter()
        losses = AverageMeter()
        recorder = Recorder('Output', MPI.COMM_WORLD.Get_size(), rank, args)

        # initialize model
        # optimizer = Adam(learning_rate=args.lr)
        optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, epsilon=1e-8)
        print('Initializing model...')
        if method == 'PGHash':
            Method = PGHash(n_labels, n_features, rank, size, 1 / size, args)
            # begin training once all devices are ready
            MPI.COMM_WORLD.Barrier()
            pg_train(rank, size, Method, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args)

        elif method == 'SLIDE':
            Method = SLIDE(n_labels, n_features, rank, size, 1 / size, args)
            # begin training once all devices are ready
            MPI.COMM_WORLD.Barrier()
            slide_train(rank, Method, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args)

        elif method == 'Regular':
            Method = Dense(n_labels, n_features, hls, sdim, num_tables, cr, rank, size, 1 / size)
            # begin training once all devices are ready
            MPI.COMM_WORLD.Barrier()
            regular_train(rank, size, Method, optimizer, train_data, test_data, losses, top1, recorder, args)

        else:
            print('ERROR: No Method Selected')
            exit()
