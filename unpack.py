import numpy as np
import tensorflow as tf
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
from mlp import SparseNeuralNetwork
from misc import compute_accuracy_lsh


class PGHash:

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type):

        self.nl = num_labels
        self.nf = num_features
        self.hls = hidden_layer_size
        self.sdim = sdim
        self.num_tables = num_tables
        self.cr = cr
        self.hash_type = hash_type
        self.used_idx = np.zeros(self.nl)
        self.num_c_layers = int(self.cr * self.nl)
        self.ci = np.arange(self.num_c_layers)
        self.final_dense = None
        self.full_layer_shapes = None
        self.full_layer_sizes = None
        self.weight_idx = (self.nf * self.hls) + self.hls + (4 * self.hls)
        self.full_idx = [range((self.weight_idx + i * self.hls), (self.weight_idx + i * self.hls + self.hls))
                         for i in self.ci]

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
            self.full_model = self.flatten_weights(self.model.get_weights())
        self.bias_start = self.full_model.size - self.nl
        self.bias_idx = self.ci + self.bias_start

    def get_model_architecture(self):
        # find shape and total elements for each layer of the resnet model
        model_weights = self.model.get_weights()
        layer_shapes = []
        layer_sizes = []
        for i in range(len(model_weights)):
            layer_shapes.append(model_weights[i].shape)
            layer_sizes.append(model_weights[i].size)
        return layer_shapes, layer_sizes

    def get_indices(self):
        self.bias_idx = self.ci + self.bias_start
        self.full_idx = [range((self.weight_idx + i * self.hls), (self.weight_idx + i * self.hls + self.hls))
                         for i in self.ci]

    def update_full_model(self, weights, biases):
        # Update this in the future to gather the start size and not know based off of fixed network
        self.full_model[self.full_idx] = weights.T
        self.full_model[self.bias_idx] = biases

    def get_final_dense(self):
        dense_shape = (self.nl, self.hls)
        end_idx = self.weight_idx + (self.hls * self.nl)
        self.final_dense = self.full_model[self.weight_idx:end_idx].reshape(dense_shape).T

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
        self.get_indices()
        # record the indices selected
        self.used_idx[self.ci] += 1
        return self.ci

    def get_new_model(self):
        # move back to the top now that we need to reset full model
        worker_layer_dims = [self.nf, self.hls, len(self.ci)]
        self.model = SparseNeuralNetwork(worker_layer_dims)
        self.layer_shapes, self.layer_sizes = self.get_model_architecture()
        # get biases
        biases = self.full_model[self.bias_idx]
        # get weights
        weights = self.full_model[self.full_idx].T
        # set new sub-model
        sub_model = np.concatenate((self.full_model[:self.weight_idx], weights.flatten(), biases.flatten()))
        new_weights = self.unflatten_weights(sub_model)
        self.model.set_weights(new_weights)
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
        self.get_new_model()
        return acc_meter.avg
