import numpy as np
import tensorflow as tf
from util.lsh import slide_lsh, dwta
from models.base import ModelHub
from util.misc import compute_accuracy_lsh
from util.network import SparseNeuralNetwork
import time
from mpi4py import MPI


class SLIDE(ModelHub):
    """
    Class to apply the SLIDE algorithm to recommender systems.
    """

    def __init__(self, num_labels, num_features, rank, size, influence, args):

        super().__init__(num_labels, num_features, args.hidden_layer_size, args.c, args.k, args.cr, rank, size,
                         influence)
        """
        Initializing the SLIDE class.
        :param num_labels: Dimensionality of labels in recommender system dataset
        :param num_features: Dimensionality of features in recommender system dataset
        :param rank: Rank of process
        :param size: Total number of processes
        :param influence: The communication weighting each process is assigned (usually is uniform)
        :param args: Remainder of arguments
        """

        # initialize parameters and lists for tables
        self.num_tables = args.num_tables
        self.gaussians = [[] for _ in range(self.num_tables)]
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

        # Two hashing methods are SLIDE with SimHash and DWTA
        if self.dwta:
            for i in range(self.num_tables):
                gaussian, hash_dict = dwta(self.final_dense, self.k)
                self.gaussians[i] = gaussian
                self.hash_dicts[i] = hash_dict
        else:
            for i in range(self.num_tables):
                gaussian, hash_dict = slide_lsh(self.final_dense, self.hls, self.k)
                self.gaussians[i] = gaussian
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
        for i in range(self.num_tables):

            # create gaussian matrix
            gaussian = self.gaussians[i]
            hash_dict = self.hash_dicts[i]

            # DWTA
            if self.dwta:
                # Apply WTA to input vector.
                selected_weights = in_layer.T[gaussian, :]
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
            # SimHash
            else:
                # Apply PG to input vector.
                transformed_layer = np.heaviside(gaussian @ in_layer.T, 0)

                # convert  data to base 2 to remove repeats
                hash_code = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))

            for j in bs_range:
                hc = hash_code[j]
                active_neurons = hash_dict[hc]
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
        # keep track of all active neurons for each device
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
        """
        Function performs averaging amongst all processes for weights which have been changed.
        :param model: Current model for each process
        :param ci: Current indices/neurons used within the model
        :return: Communication time to perform the averaging
        """

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

    def communicate(self, model, ci):
        """
        Function which allows for different patters of averaging (periodic averaging, etc.).
        :param model: Current model for each process
        :param ci: Current indices/neurons used within the model
        :param smart: Boolean to enable smart averaging (and not lazy averaging)
        :return:
        """

        # have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0
        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1 + 1) == 0:
            self.comm_iter += 1
            t = self.smart_average_vanilla(model, ci)
            comm_time += t
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return comm_time
