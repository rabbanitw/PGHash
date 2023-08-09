import numpy as np
from base import ModelHub
from misc import top1acc
from lsh import pghash_lsh, pghashd_lsh, gpu_pghash_lsh, gpu_pghashd_lsh
import torch
import time


class PGHash(ModelHub):
    """
    Class to apply our PGHash(-D) algorithm to recommender systems.
    """

    def __init__(self, num_labels, num_features, rank, size, influence, device, args):
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
        super().__init__(device, num_labels, num_features, args.hidden_layer_size, args.c, args.k, args.cr, rank, size,
                         influence)

        # initialize parameters and lists for tables
        self.num_tables = args.num_tables
        self.SB = [[] for _ in range(self.num_tables)]
        self.hash_dicts = [[] for _ in range(self.num_tables)]
        self.dwta = args.dwta
        self.device = device

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
                test_acc1 = top1acc(y_pred_test, y_batch_test)
                acc_meter.update(test_acc1, test_batch)
        else:
            test_data.shuffle(len(test_data))
            sub_test_data = test_data.take(num_batches)
            for (x_batch_test, y_batch_test) in sub_test_data:
                y_pred_test = self.big_model(x_batch_test, training=False)
                test_batch = x_batch_test.get_shape()[0]
                test_acc1 = top1acc(y_pred_test, y_batch_test)
                acc_meter.update(test_acc1, test_batch)
        return acc_meter.avg

    def test_accuracy(self, model, device, test_data_loader, running_accuracy, test_batches, epoch=False):
        running_accuracy.reset()
        j = 0
        with torch.no_grad():
            for samples, labels in test_data_loader:

                if j == test_batches and not epoch:
                    break
                j += 1

                # add data to model
                samples = samples.to(device)
                batches, n = labels.shape

                # Forward pass
                outputs = model(samples)

                # running_accuracy.update(acc / batches, batches)


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
                perm_weight = self.model.linear2.weight.t()[perm, :]
                # perm_weight = self.final_dense[perm, :]
                # run PGHash-D LSH
                hash_dict = gpu_pghashd_lsh(perm_weight, self.k)
                # save the permutation list in local memory (small memory cost) and hash tables
                self.SB[i] = perm
                self.hash_dicts[i] = hash_dict
        else:
            for i in range(self.num_tables):
                # perform PGHash LSH
                # weights = self.model.linear2.weight.detach().cpu().numpy().transpose()
                weights = self.model.linear2.weight.t()
                # SB, hash_dict = pghash_lsh(weights, self.hls, self.k, self.c)
                SB, hash_dict = gpu_pghash_lsh(self.device, weights, self.hls, self.k, self.c)
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

        # get input layer for LSH using current model
        data = data.to(self.device)
        in_layer = model.hidden_forward(data).detach().cpu().numpy()

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

        return self.ci, local_active_counter, true_neurons_bool
