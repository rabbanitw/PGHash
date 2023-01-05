import numpy as np
import tensorflow as tf
import time
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla, pghash
from mlp import SparseNeuralNetwork
from misc import compute_accuracy_lsh
from mpi4py import MPI


class PGHash:

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type,
                 rank, size, q, influence, i1, i2):

        # initialize all parameters
        self.nl = num_labels
        self.nf = num_features
        self.hls = hidden_layer_size
        self.sdim = sdim
        self.num_tables = num_tables
        self.cr = cr
        self.q = q
        self.hash_type = hash_type
        self.rank = rank
        self.size = size
        self.influence = influence
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        self.comm_iter = 0
        self.used_idx = np.zeros(self.nl)
        self.ci = np.arange(self.nl)
        self.final_dense = None
        self.full_layer_shapes = None
        self.full_layer_sizes = None

        # initialize the start of the compressed network weights and the total number of compressed labels
        self.num_c_layers = int(self.cr * self.nl)
        self.weight_idx = (self.nf * self.hls) + self.hls + (4 * self.hls)

        # initialize model
        worker_layer_dims = [self.nf, self.hls, self.num_c_layers]
        self.model = SparseNeuralNetwork(worker_layer_dims)
        self.layer_shapes, self.layer_sizes = self.get_model_architecture()

        if self.cr < 1:
            initializer = tf.keras.initializers.GlorotUniform()
            initial_final_dense = initializer(shape=(self.hls, self.nl)).numpy()
            partial_model = self.flatten_weights(self.model.get_weights())
            partial_size = partial_model.size
            missing_bias = self.nl - self.num_c_layers
            missing_weights = self.hls * (self.nl - self.num_c_layers)
            full_dense_weights = self.hls * self.nl
            self.full_model = np.zeros(partial_size + missing_weights + missing_bias)
            self.full_model[:self.weight_idx] = partial_model[:self.weight_idx]
            self.full_model[self.weight_idx:(self.weight_idx + full_dense_weights)] = initial_final_dense.T.flatten()
        elif self.cr > 1:
            print('ERROR: Compression Ratio is Greater Than 1 Which is Impossible!')
        else:
            print('No Compression Being Used')
            self.full_model = self.flatten_weights(self.model.get_weights())

        # determine where the bias index starts
        self.bias_start = self.full_model.size - self.nl
        self.bias_idx = self.ci + self.bias_start
        self.dense_shape = (self.hls, self.nl)

        # make all models start at the same initial model
        recv_buffer = np.empty_like(self.full_model)
        MPI.COMM_WORLD.Allreduce((1/self.size) * self.full_model, recv_buffer, op=MPI.SUM)
        self.full_model = recv_buffer
        self.update_model()

        # get back to smaller size
        self.ci = np.arange(self.num_c_layers)
        self.bias_idx = self.ci + self.bias_start

    def get_model_architecture(self):
        model_weights = self.model.get_weights()
        layer_shapes = []
        layer_sizes = []
        for i in range(len(model_weights)):
            layer_shapes.append(model_weights[i].shape)
            layer_sizes.append(model_weights[i].size)
        return layer_shapes, layer_sizes

    def get_final_dense(self):
        self.final_dense = self.full_model[self.weight_idx:self.bias_start].reshape(self.dense_shape)

    def get_partial_model(self):
        return self.unflatten_weights(self.full_model[:self.weight_idx])

    def flatten_weights(self, weight_list):
        return np.concatenate(list(weight_list[i].flatten() for i in range(len(weight_list))))

    def unflatten_weights(self, flat_weights):
        unflatten_model = []
        start_idx = 0
        end_idx = 0
        for i in range(len(self.layer_shapes)):
            layer_size = self.layer_sizes[i]
            end_idx += layer_size
            unflatten_model.append(flat_weights[start_idx:end_idx].reshape(self.layer_shapes[i]))
            start_idx += layer_size
        return unflatten_model

    def return_model(self):
        return self.model

    def lsh_avg_simple(self, data):

        # get weights
        self.get_final_dense()
        n = self.final_dense.shape[0]

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[-3].output,
        )
        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        ham_dists = np.zeros(self.nl)

        # run LSH to find the most important weights over the entire next Q batches
        for _ in range(self.num_tables):
            g_mat, ht = pghash(self.final_dense, self.sdim, n)
            ham_dists += pg_avg(in_layer, g_mat, ht)

        # pick just the largest differences
        avg_ham_dists = -ham_dists / (bs * self.num_tables)
        self.ci = np.sort((tf.math.top_k(avg_ham_dists, self.num_c_layers)).indices.numpy())

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci


    def lsh_avg(self, data):

        # get weights
        self.get_final_dense()
        n = self.final_dense.shape[0]

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[-3].output,
        )
        total_in_layer = feature_extractor(data).numpy()
        true_bs = int(total_in_layer.shape[0]/self.q)
        ind_per_batch = int(self.num_c_layers/self.q)
        ci = np.empty(0)

        for i in range(self.q):

            in_layer = total_in_layer[(i*true_bs):((i+1)*true_bs), :]
            ham_dists = np.zeros(self.nl)

            # run LSH to find the most important weights over the entire next Q batches
            for _ in range(self.num_tables):
                g_mat, ht = pghash(self.final_dense, self.sdim, n)
                ham_dists += pg_avg(in_layer, g_mat, ht)

            # pick just the largest differences
            avg_ham_dists = -ham_dists / (true_bs * self.num_tables)
            ci = np.union1d(np.sort((tf.math.top_k(avg_ham_dists, ind_per_batch)).indices.numpy()), ci)
        self.ci = ci

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci




    def run_lsh(self, data):

        self.get_final_dense()

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[-3].output,
        )
        in_layer = feature_extractor(data).numpy()

        # run LSH to find the most important weights
        if self.hash_type == "pg_vanilla":
            self.ci = pg_vanilla(in_layer, self.final_dense, self.sdim, self.num_tables, self.cr)
        elif self.hash_type == "pg_avg":
            self.ci = pg_avg(in_layer, self.final_dense, self.sdim, self.num_tables, self.cr)
        elif self.hash_type == "slide_vanilla":
            self.ci = slide_vanilla(in_layer, self.final_dense, self.sdim, self.num_tables, self.cr)
        elif self.hash_type == "slide_avg":
            self.ci = slide_avg(in_layer, self.final_dense, self.sdim, self.num_tables, self.cr)

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start
        # record the indices selected
        self.used_idx[self.ci] += 1
        return self.ci

    def update_full_model(self, model):
        # update full model before averaging
        weights = model.get_weights()

        w = weights[-2]
        b = weights[-1]
        self.get_final_dense()
        self.final_dense[:, self.ci] = w
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()
        self.full_model[self.bias_idx] = b
        # update the first part of the model as well!
        partial_model = self.flatten_weights(weights[:-2])
        self.full_model[:self.weight_idx] = partial_model

    def update_model(self):

        # get biases
        biases = self.full_model[self.bias_idx]
        # get weights
        self.get_final_dense()
        weights = self.final_dense[:, self.ci]
        # set new sub-model
        sub_model = np.concatenate((self.full_model[:self.weight_idx], weights.flatten(), biases.flatten()))

        new_weights = self.unflatten_weights(sub_model)
        self.model.set_weights(new_weights)

    def get_new_model(self, returnModel=True):
        # move back to the top now that we need to reset full model
        worker_layer_dims = [self.nf, self.hls, len(self.ci)]
        self.model = SparseNeuralNetwork(worker_layer_dims)
        self.layer_shapes, self.layer_sizes = self.get_model_architecture()
        self.update_model()

        if returnModel:
            return self.model

    def test_full_model(self, test_data, acc_meter):

        self.model = SparseNeuralNetwork([self.nf, self.hls, self.nl])
        self.layer_shapes, self.layer_sizes = self.get_model_architecture()
        unflatten_model = self.unflatten_weights(self.full_model)
        self.model.set_weights(unflatten_model)
        label_idx = np.arange(self.nl)
        for (x_batch_test, y_batch_test) in test_data:
            test_batch = x_batch_test.get_shape()[0]
            y_pred_test = self.model(x_batch_test, training=False)
            test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
            acc_meter.update(test_acc1, test_batch)
        self.get_new_model(returnModel=False)
        return acc_meter.avg

    def average(self, model):

        self.update_full_model(model)

        # create receiving buffer
        recv_buffer = np.empty_like(self.full_model)
        # perform averaging
        tic = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence*self.full_model, recv_buffer, op=MPI.SUM)
        toc = time.time()

        # update full model
        self.full_model = recv_buffer
        # set new sub-model
        self.update_model()
        return self.model, toc - tic

    def communicate(self, model):

        # have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0
        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:
            self.comm_iter += 1
            model, t = self.average(model)
            comm_time += t
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return model, comm_time
