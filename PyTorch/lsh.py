import numpy as np
import torch
from collections import defaultdict
import math


def pghash_lsh(weights, n, k, c):
    '''
    Compute PGHash hashing scheme.
    :param weights: Entire final layer weights
    :param n: Number of neurons
    :param k: Hash length
    :param c: Sketch dimension
    :return: Gaussian used in PGHash and buckets (dictionary) for neurons
    '''

    # create gaussian matrix=
    pg_gaussian = (1 / int(n / c)) * np.tile(np.random.normal(size=(k, c)), (1, int(np.ceil(n / c))))[:, :n]

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


def gpu_pghash_lsh(device, weights, n, k, c):
    '''
    Compute PGHash hashing scheme.
    :param weights: Entire final layer weights
    :param n: Number of neurons
    :param k: Hash length
    :param c: Sketch dimension
    :return: Gaussian used in PGHash and buckets (dictionary) for neurons
    '''

    # create gaussian matrix=
    pg_gaussian = (1 / int(n / c)) * torch.tile(torch.normal(0, 1, size=(k, c)), (1, int(math.ceil(n / c))))[:, :n]
    pg_gaussian = pg_gaussian.to(device)
    weights = weights.to(device)

    # Apply PGHash to weights.
    hash_table = torch.heaviside(pg_gaussian@weights, torch.tensor([0.]).to(device)).detach().cpu().numpy()

    # convert to base 2
    hash_table = hash_table.T.dot(1 << np.arange(hash_table.T.shape[-1]))

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_table, np.arange(len(hash_table))):
        hash_dict[k].append(v)

    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)

    return pg_gaussian.detach().cpu().numpy(), hash_dict


def slide_lsh(weights, n, k):
    '''
    Compute SLIDE hashing scheme.
    :param weights: Entire final layer weights
    :param n: Number of neurons
    :param k: Hash length
    :return: Gaussian used in SLIDE and buckets (dictionary) for neurons
    '''

    # create gaussian matrix
    slide_gaussian = np.random.normal(size=(k, n))

    # Apply Slide to weights.
    hash_table = np.heaviside(slide_gaussian@weights, 0)

    # convert to base 2
    hash_table = hash_table.T.dot(1 << np.arange(hash_table.T.shape[-1]))

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_table, np.arange(len(hash_table))):
        hash_dict[k].append(v)
    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)

    return slide_gaussian, hash_dict


def dwta(weights, k):
    """
    Performs generic DWTA hashing scheme.
    :param weights: Entire final layer weights
    :param k: Hash length
    :return: Permutation list (random coordinates selected) and buckets (dictionary) for neurons
    """

    # determine the random coordinates
    permutation = np.random.choice(weights.shape[0], k, replace=False)
    selected_weights = weights[permutation, :]
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
            constant[empty_bins] += 2 * k
            empty_bins = (hash_code == -1)
            i += 1
        hash_code += constant

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_code, np.arange(len(hash_code))):
        hash_dict[k].append(v)
    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)
    return permutation, hash_dict


def pghashd_lsh(weights, k):
    """
        Performs end of PGHash-D hashing scheme.
        :param weights: Entire final layer weights
        :param k: Hash length
        :return: Buckets (dictionary) for neurons
        """
    empty_bins = np.count_nonzero(weights, axis=0) == 0
    hash_code = np.argmax(weights, axis=0)
    # if empty bins exist, run DWTA
    if np.any(empty_bins):
        # perform DWTA
        hash_code[empty_bins] = -1
        constant = np.zeros_like(hash_code)
        i = 1
        while np.any(empty_bins):
            empty_bins_roll = np.roll(empty_bins, i)
            hash_code[empty_bins] = hash_code[empty_bins_roll]
            constant[empty_bins] += 2 * k
            empty_bins = (hash_code == -1)
            i += 1
        hash_code += constant

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_code, np.arange(len(hash_code))):
        hash_dict[k].append(v)
    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)
    return hash_dict


def gpu_pghashd_lsh(device, weights, k, slide=False):
    """
        Performs end of PGHash-D hashing scheme.
        :param weights: Entire final layer weights
        :param k: Hash length
        :return: Buckets (dictionary) for neurons
        """

    if slide:
        # determine the random coordinates
        permutation = np.random.choice(weights.shape[0], k, replace=False)
        weights = weights[permutation, :]
    else:
        permutation = None

    weights = weights.to(device)
    empty_bins = torch.count_nonzero(weights, dim=0) == 0
    empty_bins = empty_bins.detach().cpu().numpy()
    hash_code = torch.argmax(weights, dim=0).detach().cpu().numpy()
    # if empty bins exist, run DWTA
    if np.any(empty_bins):
        # perform DWTA
        hash_code[empty_bins] = -1
        constant = np.zeros_like(hash_code)
        i = 1
        while np.any(empty_bins):
            empty_bins_roll = np.roll(empty_bins, i)
            hash_code[empty_bins] = hash_code[empty_bins_roll]
            constant[empty_bins] += 2 * k
            empty_bins = (hash_code == -1)
            i += 1
        hash_code += constant

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_code, np.arange(len(hash_code))):
        hash_dict[k].append(v)
    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)
    return permutation, hash_dict


def lsh_vanilla(model, device, data, k, num_tables, SBs, hash_dicts, nl, dwta=True):
    """
    Function which performs the input hashing for each sample in a batch of data.
    :param model: Current recommender system model
    :param data: Batch of training data
    :return: Active neurons for each sample in a batch of data
    """

    # get input layer for LSH using current model
    with torch.no_grad():
        in_layer = model.hidden_forward(data)
        in_layer = in_layer# .detach().cpu().numpy()

    # find batch size and initialize parameters
    bs = in_layer.shape[0]
    bs_range = np.arange(bs)
    local_active_counter = [np.zeros(nl, dtype=bool) for _ in range(bs)]
    global_active_counter = np.zeros(nl, dtype=bool)
    full_size = np.arange(nl)
    prev_global = None
    unique = 0
    num_c_layers = nl

    # run through the prescribed number of tables to find exact matches (vanilla) which are marked as active neurons
    for i in range(num_tables):
        # load gaussian (or SB) matrix
        SB = SBs[i]
        hash_dict = hash_dicts[i]

        # PGHash-D hashing style
        if dwta:
            # Apply WTA to input vector.
            selected_weights = in_layer.T[SB, :]
            empty_bins = torch.count_nonzero(selected_weights, dim=0) == 0
            hash_code = torch.argmax(selected_weights, dim=0).detach().cpu().numpy()
            empty_bins = empty_bins.detach().cpu().numpy()
            # if empty bins exist, run DWTA
            if np.any(empty_bins):
                # perform DWTA
                hash_code[empty_bins] = -1
                constant = np.zeros_like(hash_code)
                i = 1
                while np.any(empty_bins):
                    empty_bins_roll = np.roll(empty_bins, i)
                    hash_code[empty_bins] = hash_code[empty_bins_roll]
                    constant[empty_bins] += 2 * k
                    empty_bins = (hash_code == -1)
                    i += 1
                hash_code += constant

        # PGHash hashing style
        else:
            # apply PG Gaussian to input vector
            transformed_layer = torch.heaviside(torch.from_numpy(SB).to(device) @ in_layer.T,
                                                torch.tensor(0.)).detach().cpu().numpy()

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
        if unique >= num_c_layers:
            break
        else:
            # store the previous list of total neurons in the case that it can be used if the next list is over the
            # total number of neurons required (and can be randomly shaven down)
            prev_global = np.copy(global_active_counter)

    # remove selected neurons (in a smart way)
    gap = unique - num_c_layers
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
        ci = full_size[global_active_counter]
        true_neurons_bool = global_active_counter
    else:
        # in order to ensure the entire model is used (due to TF issues) we classify the true neurons and fill
        # the rest with "fake" neurons which won't be back propagated on
        remaining_neurons = full_size[np.logical_not(global_active_counter)]
        true_neurons_bool = np.copy(global_active_counter)
        fake_neurons = remaining_neurons[:-gap]
        global_active_counter[fake_neurons] = True
        ci = full_size[global_active_counter]
        for k in bs_range:
            # select only active neurons for this sample
            local_active_counter[k] = full_size[local_active_counter[k]]

    return ci, local_active_counter, true_neurons_bool
