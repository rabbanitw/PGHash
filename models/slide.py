import numpy as np
import tensorflow as tf
from lsh import slide, slide_hashtable
from models.base import ModelHub
from util.misc import compute_accuracy_lsh
from util.mlp import SparseNeuralNetwork


class SLIDE(ModelHub):

    def __init__(self, num_labels, num_features, rank, size, influence, args):

        super().__init__(num_labels, num_features, args.hidden_layer_size, args.sdim, args.num_tables, args.cr, rank,
                         size, args.q, influence)

        self.gaussian_mats = None
        self.hash_dicts = []

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

    def test_full_model(self, test_data, acc_meter):
        self.big_model.set_weights(self.unflatten_weights_big(self.full_model))
        label_idx = np.arange(self.nl)
        for (x_batch_test, y_batch_test) in test_data:
            test_batch = x_batch_test.get_shape()[0]
            y_pred_test = self.big_model(x_batch_test, training=False)
            test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
            acc_meter.update(test_acc1, test_batch)
        return acc_meter.avg

    def lsh_get_hash(self):

        # get weights
        self.get_final_dense()
        n = self.final_dense.shape[0]

        gaussian_mats = None

        # determine all the hash tables and gaussian matrices
        for i in range(self.num_tables):
            g_mat, ht_dict = slide_hashtable(self.final_dense, n, self.sdim)

            if i == 0:
                gaussian_mats = g_mat
            else:
                gaussian_mats = np.vstack((gaussian_mats, g_mat))

            self.hash_dicts.append(ht_dict)

        self.gaussian_mats = gaussian_mats

    def lsh(self, data, union=True, num_random_table=50):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[2].output,  # this is the post relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        cur_idx = [i for i in range(bs)]

        if union:
            table_idx = np.random.choice(self.num_tables, num_random_table, replace=False)
            for i in range(num_random_table):
                idx = table_idx[i]
                cur_gauss = self.gaussian_mats[(idx * self.sdim):((idx + 1) * self.sdim), :]
                cur_ht_dict = self.hash_dicts[idx]
                hash_idxs = slide(in_layer, cur_gauss, cur_ht_dict)
                for j in range(bs):
                    if i == 0:
                        cur_idx[j] = hash_idxs[j]
                    else:
                        cur_idx[j] = np.union1d(cur_idx[j], hash_idxs[j])

        else:
            prev_cur_idx = [i for i in range(bs)]
            gap_idx = -np.ones(bs, dtype=np.int64)

            for i in range(self.num_tables):
                cur_gauss = self.gaussian_mats[(i*self.sdim):((i+1)*self.sdim), :]
                cur_ht_dict = self.hash_dicts[i]
                hash_idxs = slide(in_layer, cur_gauss, cur_ht_dict)
                for j in range(bs):

                    # if already filled, then skip
                    if gap_idx[j] == 0:
                        continue

                    if i == 0:
                        cur_idx[j] = hash_idxs[j]
                    else:
                        cur_idx[j] = np.intersect1d(cur_idx[j], hash_idxs[j])
                    gap_idx[j] = int(self.num_c_layers - len(cur_idx[j]))

                    # if we have not filled enough, then randomly select indices from the previous cur_idx to fill the gap
                    if gap_idx[j] > 0:
                        if i == 0:
                            prev_dropped_idx = np.setdiff1d(np.arange(self.nl), cur_idx[j])
                        else:
                            prev_dropped_idx = np.setdiff1d(prev_cur_idx[j], cur_idx[j])
                        cur_idx[j] = np.union1d(cur_idx[j], np.random.choice(prev_dropped_idx, gap_idx[j], replace=False))
                        gap_idx[j] = 0
                        continue

                    prev_cur_idx[j] = cur_idx[j]

                    # if X tables is not enough, take a random choice of the leftover (very unlikely)
                    if i == self.num_tables - 1 and gap_idx[j] < 0:
                        cur_idx[j] = np.random.choice(cur_idx[j], self.num_c_layers)

                # if all(gap_idx == 0):
                if all(gap_idx > 0):
                    break

        return cur_idx

    def update(self, prev_weights, non_active_indices):
        # update full model before averaging
        w = prev_weights[-2]
        b = prev_weights[-1]
        self.get_final_dense()
        self.final_dense[:, non_active_indices] = w[:, non_active_indices]
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()
        self.full_model[non_active_indices + self.bias_start] = b[non_active_indices]

        # update the first part of the model as well!
        # partial_model = self.flatten_weights(prev_weights[:-2])
        # self.full_model[:self.weight_idx] = partial_model

        # set new weights
        new_weights = self.unflatten_weights(self.full_model)
        self.model.set_weights(new_weights)

        return self.model