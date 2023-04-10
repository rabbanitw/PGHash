import numpy as np
from collections import defaultdict


def pg_hashtable(weights, n, c, sdim):
    '''
    compute hashing
    :param vectors:
    :param n:
    :param sdim:
    :return:
    '''

    # create gaussian matrix
    pg_gaussian = (1/int(n/sdim))*np.tile(np.random.normal(size=(c, sdim)), (1, int(np.ceil(n/sdim))))[:, :n]

    # Apply PGHash to weights.
    hash_table = np.heaviside(pg_gaussian@weights, 0)

    # convert to base 2
    hash_table = hash_table.T.dot(1 << np.arange(hash_table.T.shape[-1]))

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_table, np.arange(len(hash_table))):
        hash_dict[k].append(v)

    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)

    return pg_gaussian, hash_dict


def rehash(num_tables):
    for i in range(num_tables):
        gaussian, hash_dict = pg_hashtable(self.final_dense, self.hls, self.c, self.sdim)
        self.gaussians[i] = gaussian
        self.hash_dicts[i] = hash_dict


def lsh_vanilla(input, label, num_labels):

    bs = input.shape[0]
    bs_range = np.arange(bs)
    local_active_counter = [np.zeros(num_labels, dtype=bool) for _ in range(bs)]
    global_active_counter = np.zeros(num_labels, dtype=bool)
    full_size = np.arange(num_labels)
    prev_global = None
    unique = 0

    for i in range(self.num_tables):
        # load gaussian matrix
        gaussian = self.gaussians[i]
        hash_dict = self.hash_dicts[i]

        # Apply PG to input vector.
        transformed_layer = np.heaviside(gaussian @ input.T, 0)

        # convert  data to base 2 to remove repeats
        base2_hash = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))
        for j in bs_range:
            hc = base2_hash[j]
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

        neurons_per_sample = []
        max_neurons = 0
        for k in bs_range:
            # shave off deactivated neurons
            lac = local_active_counter[k] * global_active_counter
            neurons = np.count_nonzero(lac)
            neurons_per_sample.append(neurons)
            # select only active neurons for this sample
            local_active_counter[k] = full_size[lac]
            if neurons > max_neurons:
                max_neurons = neurons

    else:
        neurons_per_sample = []
        max_neurons = 0
        for k in bs_range:
            # select only active neurons for this sample
            lac = local_active_counter[k]
            neurons = np.count_nonzero(lac)
            neurons_per_sample.append(neurons)
            local_active_counter[k] = full_size[lac]
            if neurons > max_neurons:
                max_neurons = neurons

    padded_active_neurons = num_labels * np.ones((bs, max_neurons))
    new_label = np.zeros((bs, max_neurons))
    for k in bs_range:
        lac = local_active_counter[k]
        nps = neurons_per_sample[k]
        padded_active_neurons[k, :nps] = lac
        new_label[k, :nps] = label[k, lac]

    return padded_active_neurons, new_label, global_active_counter
