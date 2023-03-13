import numpy as np
import tensorflow as tf
from mpi4py import MPI
from models.base import ModelHub
from util.mlp import SparseNeuralNetwork
from util.misc import compute_accuracy_lsh
import time
from lsh import pg_hashtable


def get_unique_N(iterable, N):
    """Yields (in order) the first N unique elements of iterable.
    Might yield less if data too short."""
    seen = set()
    for e in iterable:
        if e in seen:
            continue
        seen.add(e)
        yield e
        if len(seen) == N:
            return


class PGHash(ModelHub):

    def __init__(self, num_labels, num_features, rank, size, influence, args):

        super().__init__(num_labels, num_features, args.hidden_layer_size, args.sdim, args.cr, rank, size, args.q,
                         influence)

        self.num_tables = args.num_tables
        self.slide = args.slide
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
            gaussian, hash_dict = pg_hashtable(self.final_dense, self.hls, self.sdim)
            self.gaussians[i] = gaussian
            self.hash_dicts[i] = hash_dict

    def lsh_vanilla(self, model, data, sparse_rehash=True):

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
            if not sparse_rehash:
                gaussian, hash_dict = pg_hashtable(self.final_dense, self.hls, self.sdim)
            else:
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

    def exchange_idx_vanilla(self, bool_idx):
        t = time.time()
        dev_bools = np.empty((self.size, self.nl), dtype=np.bool_)
        MPI.COMM_WORLD.Allgather(bool_idx, dev_bools)
        count = np.sum(dev_bools, axis=0)
        self.count = count[count > 0]
        total_active = np.zeros(self.nl, dtype=bool)
        self.device_idxs = []
        for dev in range(self.size):
            dev_bool = dev_bools[dev, :]
            total_active += dev_bool
            self.device_idxs.append(self.full_size[dev_bool])
        self.unique = self.full_size[total_active]
        self.unique_len = len(self.unique)
        self.unique_idx = np.empty(self.nl, dtype=np.int64)
        self.unique_idx[self.unique] = np.arange(self.unique_len)
        return time.time()-t

    def smart_average_vanilla(self, model, ci):

        # update the model
        self.update_full_model(model)
        comm_time = 0

        # create receiving buffer for first dense layer
        recv_first_layer = np.empty_like(self.full_model[:self.weight_idx])
        # Allreduce first layer of the network
        MPI.COMM_WORLD.Barrier()
        t = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence * self.full_model[:self.weight_idx], recv_first_layer, op=MPI.SUM)
        comm_time += (time.time() - t)
        # update first layer
        self.full_model[:self.weight_idx] = recv_first_layer

        # prepare the layers and biases to send
        send_final_layer = self.final_dense[:, ci]
        send_final_bias = self.full_model[self.bias_start + ci]

        updated_final_layer = np.zeros((self.hls, self.unique_len))
        updated_final_bias = np.zeros(self.unique_len)
        updated_final_layer[:, self.unique_idx[self.device_idxs[self.rank]]] += send_final_layer
        updated_final_bias[self.unique_idx[self.device_idxs[self.rank]]] += send_final_bias

        send_buf = np.concatenate((updated_final_layer.flatten(), updated_final_bias))
        recv_buf = np.empty((self.hls+1) * self.unique_len)
        t = time.time()
        MPI.COMM_WORLD.Allreduce(send_buf, recv_buf, op=MPI.SUM)
        comm_time += (time.time() - t)

        updated_final_layer = recv_buf[:-self.unique_len].reshape(self.hls, self.unique_len) / self.count
        updated_final_bias = recv_buf[-self.unique_len:] / self.count

        # update the full model
        # set biases
        self.full_model[self.bias_start + self.unique] = updated_final_bias
        # set weights
        self.final_dense[:, self.unique] = updated_final_layer
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()

        # update the sub-architecture
        self.update_model()

        return comm_time

    def simple_average(self, model):

        self.update_full_model(model)

        # create receiving buffer
        recv_buffer = np.empty_like(self.full_model)
        # perform averaging
        tic = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence * self.full_model, recv_buffer, op=MPI.SUM)
        toc = time.time()

        # update full model
        self.full_model = recv_buffer
        # set new sub-model
        self.update_model()
        return toc - tic

    def communicate(self, model, ci, smart=True):

        # have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0
        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1 + 1) == 0:
            self.comm_iter += 1
            if smart:
                t = self.smart_average_vanilla(model, ci)
            else:
                t = self.simple_average(model)
            comm_time += t
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return comm_time

    # Below are other versions of LSH (non-vanilla)

    def lsh_hamming(self, model, data, num_tables=5, cutoff=2):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        bs_range = np.arange(bs)
        local_active_counter = [np.zeros(self.nl, dtype=bool) for _ in range(bs)]
        selected_neuron_list = [[] for _ in range(bs)]
        global_active_counter = np.zeros(self.nl, dtype=bool)
        prev_global = None
        thresh = 1

        for i in range(num_tables):

            # create gaussian matrix
            pg_gaussian = (1 / int(self.hls / self.sdim)) * np.tile(np.random.normal(size=(self.sdim, self.sdim)),
                                                                    int(np.ceil(self.hls / self.sdim)))[:, :self.hls]

            # pg_gaussian = np.random.normal(size=(self.sdim, self.hls))

            # Apply PGHash to weights.
            hash_table = np.heaviside(pg_gaussian @ self.final_dense, 0)

            # Apply PG to input vector.
            transformed_layer = np.heaviside(pg_gaussian @ in_layer.T, 0)

            # convert data to base 2 to remove repeats
            base2_hash = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))

            for j in bs_range:
                hc = base2_hash[j]
                if hc in base2_hash[:j]:
                    # if hamming distance is already computed
                    h_idx = np.where(base2_hash[:j] == hc)
                    h_idx = h_idx[0][0]
                    selected_neurons = selected_neuron_list[h_idx]
                    local_active_counter[j][selected_neurons] = True
                else:
                    # compute hamming distances
                    hamm_dists = np.count_nonzero(hash_table != transformed_layer[:, j, np.newaxis], axis=0)
                    selected_neurons = self.full_size[hamm_dists < thresh]  # choose 2 for now
                    selected_neuron_list[j] = selected_neurons
                    local_active_counter[j][selected_neurons] = True
                    global_active_counter[selected_neurons] = True

            # WHAT WE CAN ALSO DO IS BUMP UP THE VALUE OF < FOR HAMMING UP UNTIL A CERTAIN AMOUNT AND THEN RERUN TABLE

            unique = np.count_nonzero(global_active_counter)
            if unique >= self.num_c_layers:
                break
            else:
                prev_global = np.copy(global_active_counter)
                # after 2 tables if haven't filled then increase the match size
                if i == 1:
                    thresh = cutoff
                elif i == num_tables-2:
                    thresh = int(self.sdim/2)

        # remove selected neurons (in a smart way)
        gap = unique - self.num_c_layers

        if gap > 0:
            if prev_global is None:
                p = global_active_counter / unique
            else:
                # remove most recent selected neurons if multiple tables are used
                change = global_active_counter != prev_global
                p = change / np.count_nonzero(change)

            deactivate = np.random.choice(self.full_size, size=gap, replace=False, p=p)
            global_active_counter[deactivate] = False

        for k in bs_range:
            # shave off deactivated neurons
            local_active_counter[k] = local_active_counter[k] * global_active_counter
            # select only active neurons for this sample
            local_active_counter[k] = self.full_size[local_active_counter[k]]

        if gap >= 0:
            self.ci = self.full_size[global_active_counter]
            fake_neurons = None
            true_neurons_bool = global_active_counter
        else:
            remaining_neurons = self.full_size[np.logical_not(global_active_counter)]
            true_neurons_bool = np.copy(global_active_counter)
            fake_neurons = remaining_neurons[:-gap]
            global_active_counter[fake_neurons] = True
            self.ci = self.full_size[global_active_counter]

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, local_active_counter, true_neurons_bool, fake_neurons

    def lsh_hamming_opt(self, model, data):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        bs_range = np.arange(bs)
        cur_idx = np.empty((self.num_c_layers, bs), dtype=np.int)

        # create gaussian matrix
        pg_gaussian = (1 / int(self.hls / self.sdim)) * np.tile(np.random.normal(size=(self.sdim, self.sdim)),
                                                                int(np.ceil(self.hls / self.sdim)))[:, :self.hls]

        # pg_gaussian = np.random.normal(size=(self.sdim, self.hls))

        # Apply PGHash to weights.
        hash_table = np.heaviside(pg_gaussian @ self.final_dense, 0)

        # Apply PG to input vector.
        transformed_layer = np.heaviside(pg_gaussian @ in_layer.T, 0)

        # convert data to base 2 to remove repeats
        base2_hash = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))

        unique = []
        for j in bs_range:
            hc = base2_hash[j]
            if hc in base2_hash[:j]:
                # if hamming distance is already computed
                h_idx = np.where(base2_hash[:j] == hc)
                h_idx = h_idx[0][0]
                cur_idx[:, j] = cur_idx[:, h_idx]

            else:
                # compute hamming distances
                hamm_dists = np.count_nonzero(hash_table != transformed_layer[:, j, np.newaxis], axis=0)
                # compute the topk closest average hamming distances to neuron
                cur_idx[:, j] = np.argsort(hamm_dists)[:self.num_c_layers]
                # cur_idx[j] = topk_by_partition(hamm_dists, self.num_c_layers)
                unique.append(j)

        # make sure transposed to get top hamming distance for each sample (maybe should shuffle samples before too)
        cur_idx_1d = cur_idx[:, unique].flatten()

        # first grab the known unique values
        k = list(get_unique_N(cur_idx_1d, self.num_c_layers))
        self.ci = np.sort(k)

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, None

'''
        def exchange_idx(self):
        t = time.time()
        if self.rank == 0:
            self.device_idxs = np.empty((self.size, self.num_c_layers), dtype=np.int32)
        send_buf = -1*np.ones(self.num_c_layers, dtype=np.int32)
        send_buf[:len(self.ci)] = self.ci
        MPI.COMM_WORLD.Gather(send_buf, self.device_idxs, root=0)
        if self.rank == 0:
            temp = []
            for dev in range(self.size):
                dev_idx = self.device_idxs[dev, :]
                temp.append(dev_idx[dev_idx != -1])
            self.device_idxs = temp
            self.unique, self.count = np.unique(np.concatenate(temp, dtype=np.int32), return_counts=True)
            self.unique_len = len(self.unique)
            self.unique_idx = np.empty(self.nl, dtype=np.int64)
            self.unique_idx[self.unique] = np.arange(self.unique_len)
            MPI.COMM_WORLD.Bcast(np.array([self.unique_len]), root=0)
            MPI.COMM_WORLD.Bcast(self.unique, root=0)
        else:
            data = np.empty(1, dtype=np.int64)
            MPI.COMM_WORLD.Bcast(data, root=0)
            self.unique_len = data[0]
            data = np.empty(self.unique_len, dtype=np.int32)
            MPI.COMM_WORLD.Bcast(data, root=0)
            self.unique = data
        return time.time()-t
'''