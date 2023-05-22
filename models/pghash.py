import numpy as np
import tensorflow as tf
from mpi4py import MPI
from models.base import ModelHub
from util.network import SparseNeuralNetwork
from util.misc import compute_accuracy_lsh
import time
from lsh import pghash_lsh, pghashd_lsh


class PGHash(ModelHub):
    """
    Class to apply our PGHash(-D) algorithm to recommender systems.
    """

    def __init__(self, num_labels, num_features, rank, size, influence, args):
        """
        Initializing the PGHash class.
        :param num_labels: Dimensionality of labels in recommender system dataset
        :param num_features: Dimensionality of features in recommender system dataset
        :param rank: Rank of process
        :param size: Total number of processes
        :param influence: The communication weighting each process is assigned (usually is uniform)
        :param args: Remainder of arguments
        """

        # inherit the general Base class (which deals with updating the model dynamically)
        super().__init__(num_labels, num_features, args.hidden_layer_size, args.c, args.k, args.cr, rank, size,
                         influence)

        # initialize parameters and lists for tables
        self.num_tables = args.num_tables
        self.SB = [[] for _ in range(self.num_tables)]
        self.hash_dicts = [[] for _ in range(self.num_tables)]
        self.dwta = args.dwta

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

    def test_full_model(self, test_data, acc_meter, epoch_test=True, num_batches=30):
        """
        Function which performs testing on the
        :param test_data: Test data for recommender system
        :param acc_meter: Accuracy meter which stores the average accuracies
        :param epoch_test: Boolean dictating if testing is done on entire or subset of test data
        :param num_batches: Number of batches of the test set used to evaluate the accuracy (to reduce comp. costs)
        :return: Average test accuracy over either the entire test set or a subset of batches
        """

        # load the entire model to test on the root process
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
            sub_test_data = test_data.take(num_batches)
            for (x_batch_test, y_batch_test) in sub_test_data:
                y_pred_test = self.big_model(x_batch_test, training=False)
                test_batch = x_batch_test.get_shape()[0]
                test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
                acc_meter.update(test_acc1, test_batch)
        return acc_meter.avg

    def rehash(self):
        """
        Function which performs the weight hashing every X iterations for PGHash.
        :return: A new set of hash codes (and subsequent buckets) for the final layer weights
        """

        # Two hashing methods are PGHash and PGHash-D (DWTA variant)
        if self.dwta:
            # create list of possible coordinates to randomly select for DWTA
            potential_indices = np.arange(self.hls)
            for i in range(self.num_tables):
                # server randomly selects coordinates for DWTA (stored as permutation list), vector is dimension c
                indices = np.random.choice(potential_indices, self.c, replace=False)
                # process selects a k-subset of the c coordinates to use for on-device DWTA
                perm = np.random.choice(indices, self.k, replace=False)
                # index the k coordinates for each neuron (size is k x n)
                perm_weight = self.final_dense[perm, :]
                # run PGHash-D LSH
                hash_dict = pghashd_lsh(perm_weight, self.k)
                # save the permutation list in local memory (small memory cost) and hash tables
                self.SB[i] = perm
                self.hash_dicts[i] = hash_dict
        else:
            for i in range(self.num_tables):
                # perform PGHash LSH
                SB, hash_dict = pghash_lsh(self.final_dense, self.hls, self.k, self.c)
                # save gaussian and hash tables
                # when rehashing is performed every step, these cn be immediately discarded
                self.SB[i] = SB
                self.hash_dicts[i] = hash_dict

    def lsh_vanilla(self, model, data):
        """
        Function which performs the input hashing for each sample in a batch of data.
        :param model: Current recommender system model
        :param data: Batch of training data
        :return: Active neurons for each sample in a batch of data
        """

        # initialize half model which will spit out the input to the final dense layer
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
        )

        # get input layer for LSH using current model
        in_layer = feature_extractor(data).numpy()

        # find batch size and initialize parameters
        bs = in_layer.shape[0]
        bs_range = np.arange(bs)
        local_active_counter = [np.zeros(self.nl, dtype=bool) for _ in range(bs)]
        global_active_counter = np.zeros(self.nl, dtype=bool)
        full_size = np.arange(self.nl)
        prev_global = None
        unique = 0

        # run through the prescribed number of tables to find exact matches (vanilla) which are marked as active neurons
        for i in range(self.num_tables):
            # load gaussian (or SB) matrix
            SB = self.SB[i]
            hash_dict = self.hash_dicts[i]

            # PGHash-D hashing style
            if self.dwta:
                # Apply WTA to input vector.
                selected_weights = in_layer.T[SB, :]
                empty_bins = np.count_nonzero(selected_weights, axis=0) == 0
                hash_code = np.argmax(selected_weights, axis=0)
                # if empty bins exist, run DWTA
                if np.any(empty_bins):
                    # perform DWTA
                    hash_code[empty_bins] = -1
                    constant = np.zeros_like(hash_code)
                    i = 1
                    while np.any(empty_bins):
                        empty_bins_roll = np.roll(empty_bins, i)
                        hash_code[empty_bins] = hash_code[empty_bins_roll]
                        constant[empty_bins] += 2 * self.k
                        empty_bins = (hash_code == -1)
                        i += 1
                    hash_code += constant

            # PGHash hashing style
            else:
                # apply PG Gaussian to input vector
                transformed_layer = np.heaviside(SB @ in_layer.T, 0)

                # convert data to base 2 to remove repeats
                hash_code = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))

            # after computing hash codes for each sample, loop over the samples and match them to neurons
            for j in bs_range:
                # find current sample hash code
                hc = hash_code[j]
                # determine neurons which have the same hash code
                active_neurons = hash_dict[hc]
                # mark these neurons as active for the sample as well as the global counter
                local_active_counter[j][active_neurons] = True
                global_active_counter[active_neurons] = True

            # compute how many neurons are active across the ENTIRE batch
            unique = np.count_nonzero(global_active_counter)

            # once the prescribed total number of neurons are reached, end LSH
            if unique >= self.num_c_layers:
                break
            else:
                # store the previous list of total neurons in the case that it can be used if the next list is over the
                # total number of neurons required (and can be randomly shaven down)
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

            # randomly select neurons from most recent table to deactivate
            deactivate = np.random.choice(full_size, size=gap, replace=False, p=p)
            global_active_counter[deactivate] = False

            for k in bs_range:
                # shave off deactivated neurons
                lac = local_active_counter[k] * global_active_counter
                # select only active neurons for this sample
                local_active_counter[k] = full_size[lac]

            # set the current active neurons across the ENTIRE batch
            self.ci = full_size[global_active_counter]
            true_neurons_bool = global_active_counter
        else:
            # in order to ensure the entire model is used (due to TF issues) we classify the true neurons and fill
            # the rest with "fake" neurons which won't be back propagated on
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

        return self.ci, local_active_counter, true_neurons_bool

    def exchange_idx_vanilla(self, bool_idx):
        """
        Function sends the processes the indices each process uses in order to average correctly
        :param bool_idx: boolean list of which neurons are active for each sample
        :return: Time taken to perform this function
        """
        t = time.time()
        dev_bools = np.empty((self.size, self.nl), dtype=np.bool_)
        # send boolean list to all other processes (equivalent to central server)
        MPI.COMM_WORLD.Allgather(bool_idx, dev_bools)
        # count how many processes had each neuron active
        count = np.sum(dev_bools, axis=0)
        # only care about the neurons which were updated (non-updated neurons are not averaged)
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

    # Below are other versions of LSH (non-vanilla)... THESE ARE UNUSED IN OUR CODE BUT CAN BE IMPLEMENTED
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
        pg_gaussian = (1 / int(self.hls / self.c)) * np.tile(np.random.normal(size=(self.c, self.c)),
                                                                int(np.ceil(self.hls / self.c)))[:, :self.hls]

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
