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
