import numpy as np
import tensorflow as tf
from util.mlp import SparseNeuralNetwork
from mpi4py import MPI


class ModelHub:

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, rank, size, q, influence,
                 i1=0, i2=1):

        # initialize all parameters
        self.nl = num_labels
        self.nf = num_features
        self.hls = hidden_layer_size
        self.sdim = sdim
        self.num_tables = num_tables
        self.cr = cr
        self.q = q
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
        self.device_idxs = None
        self.unique = None
        self.unique_len = None
        self.unique_idx = None
        self.count = None
        self.big_model = None

        # initialize the start of the compressed network weights and the total number of compressed labels
        self.num_c_layers = int(self.cr * self.nl)
        self.weight_idx = (self.nf * self.hls) + self.hls

        # initialize compressed model
        worker_layer_dims = [self.nf, self.hls, self.num_c_layers]
        self.model = SparseNeuralNetwork(worker_layer_dims)
        self.layer_shapes, self.layer_sizes = self.get_model_architecture()

        # initialize full model in numpy form which is used for communication between devices
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
            exit()
        else:
            print('No Compression Being Used')
            self.full_model = self.flatten_weights(self.model.get_weights())

        # determine where the bias index starts
        self.bias_start = self.full_model.size - self.nl
        self.bias_idx = self.ci + self.bias_start
        self.dense_shape = (self.hls, self.nl)

        # make all devices start at same initial model
        self.sync_models()

        # get back to smaller size
        self.ci = np.sort(np.random.choice(self.nl, self.num_c_layers, replace=False))
        self.bias_idx = self.ci + self.bias_start

    def get_model_architecture(self):
        model_weights = self.model.get_weights()
        layer_shapes = []
        layer_sizes = []
        for i in range(len(model_weights)):
            layer_shapes.append(model_weights[i].shape)
            layer_sizes.append(model_weights[i].size)
        return layer_shapes, layer_sizes

    def sync_models(self):
        # make all models start at the same initial model
        recv_buffer = np.empty_like(self.full_model)
        MPI.COMM_WORLD.Allreduce((1 / self.size) * self.full_model, recv_buffer, op=MPI.SUM)
        self.full_model = recv_buffer
        self.update_model()

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

    def unflatten_weights_big(self, flat_weights):
        unflatten_model = []
        start_idx = 0
        end_idx = 0
        for i in range(len(self.full_layer_shapes)):
            layer_size = self.full_layer_sizes[i]
            end_idx += layer_size
            unflatten_model.append(flat_weights[start_idx:end_idx].reshape(self.full_layer_shapes[i]))
            start_idx += layer_size
        return unflatten_model

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

    def update_model(self, return_model=False):

        # get biases
        biases = self.full_model[self.bias_idx]
        # get weights
        self.get_final_dense()
        weights = self.final_dense[:, self.ci]
        # set new sub-model
        sub_model = np.concatenate((self.full_model[:self.weight_idx], weights.flatten(), biases.flatten()))

        new_weights = self.unflatten_weights(sub_model)
        self.model.set_weights(new_weights)
        if return_model:
            return self.model
