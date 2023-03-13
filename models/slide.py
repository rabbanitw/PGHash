import numpy as np
import tensorflow as tf
from lsh import slide_hashtable
from models.base import ModelHub
from util.misc import compute_accuracy_lsh
from util.mlp import SparseNeuralNetwork


class SLIDE(ModelHub):

    def __init__(self, num_labels, num_features, rank, size, influence, args):

        super().__init__(num_labels, num_features, args.hidden_layer_size, args.sdim, args.cr, rank, size, influence)

        self.num_tables = args.num_tables
        self.gaussians = [[] for _ in range(self.num_tables)]
        self.hash_dicts = [[] for _ in range(self.num_tables)]

        # initialize full model for the root device for testing accuracy
        if self.rank == 0:
            self.big_model = SparseNeuralNetwork([self.nf, self.hls, self.nl])
            mw = self.big_model.get_weights()
            layer_shapes = []
            layer_sizes = []
            for i in range(len(mw)):
                layer_shapes.append(mw[i].shape)
                layer_sizes.append(mw[i].size)
            self.full_layer_sizes = layer_sizes
            self.full_layer_shapes = layer_shapes

            # set big model weights
            self.big_model.set_weights(self.unflatten_weights_big(self.full_model))

    def test_full_model(self, test_data, acc_meter, epoch_test=True):
        self.big_model.set_weights(self.unflatten_weights_big(self.full_model))
        label_idx = np.arange(self.nl)
        if epoch_test:
            for (x_batch_test, y_batch_test) in test_data:
                y_pred_test = self.big_model(x_batch_test, training=False)
                test_batch = x_batch_test.get_shape()[0]
                test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
                acc_meter.update(test_acc1, test_batch)
        else:
            test_data.shuffle(len(test_data))
            sub_test_data = test_data.take(30)
            for (x_batch_test, y_batch_test) in sub_test_data:
                y_pred_test = self.big_model(x_batch_test, training=False)
                test_batch = x_batch_test.get_shape()[0]
                test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
                acc_meter.update(test_acc1, test_batch)
        return acc_meter.avg

    def rehash(self):
        for i in range(self.num_tables):
            gaussian, hash_dict = slide_hashtable(self.final_dense, self.hls, self.sdim)
            self.gaussians[i] = gaussian
            self.hash_dicts[i] = hash_dict

    def lsh_vanilla(self, model, data):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        bs_range = np.arange(bs)
        local_active_counter = [np.zeros(self.nl, dtype=bool) for _ in range(bs)]
        global_active_counter = np.zeros(self.nl, dtype=bool)
        full_size = np.arange(self.nl)
        prev_global = None
        unique = 0

        for i in range(self.num_tables):

            # create gaussian matrix
            gaussian = self.gaussians[i]
            hash_dict = self.hash_dicts[i]

            # Apply PG to input vector.
            transformed_layer = np.heaviside(gaussian @ in_layer.T, 0)

            # convert  data to base 2 to remove repeats
            base2_hash = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))

            for j in bs_range:
                hc = base2_hash[j]
                active_neurons = hash_dict[hc]
                # local_active_counter[j][active_neurons] += 1
                local_active_counter[j][active_neurons] = True
                global_active_counter[active_neurons] = True

            unique = np.count_nonzero(global_active_counter)
            if unique >= self.num_c_layers:
                break
            else:
                prev_global = np.copy(global_active_counter)

        # remove selected neurons (in a smart way)
        gap = unique - self.num_c_layers
        if gap > 0:
            if prev_global is None:
                p = global_active_counter / unique
            else:
                # remove most recent selected neurons if multiple tables are used
                change = global_active_counter != prev_global
                p = change / np.count_nonzero(change)

            deactivate = np.random.choice(full_size, size=gap, replace=False, p=p)
            global_active_counter[deactivate] = False

            for k in bs_range:
                # shave off deactivated neurons
                local_active_counter[k] = local_active_counter[k] * global_active_counter
                # select only active neurons for this sample
                local_active_counter[k] = full_size[local_active_counter[k]]

            self.ci = full_size[global_active_counter]
            fake_neurons = []
            true_neurons_bool = global_active_counter
        else:
            remaining_neurons = full_size[np.logical_not(global_active_counter)]
            true_neurons_bool = np.copy(global_active_counter)
            fake_neurons = remaining_neurons[:-gap]
            global_active_counter[fake_neurons] = True
            self.ci = full_size[global_active_counter]
            for k in bs_range:
                # select only active neurons for this sample
                local_active_counter[k] = full_size[local_active_counter[k]]

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, local_active_counter, true_neurons_bool, fake_neurons
