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

    # create gaussian matrix=
    pg_gaussian = (1/int(n/sdim))*np.tile(np.random.normal(size=(c, sdim)), (1, int(np.ceil(n/sdim))))[:, :n]
    # pg_gaussian = np.random.normal(size=(sdim, n))=

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

    '''
    # convert to base 2
    hash_table = hash_table.T.dot(1 << np.arange(hash_table.T.shape[-1]))

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_table, np.arange(len(hash_table))):
        hash_dict[k].append(v)
    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)
    '''

    return pg_gaussian, hash_dict


def slide_hashtable(weights, n, c):

    # create gaussian matrix
    slide_gaussian = np.random.normal(size=(c, n))

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


def wta(weights, c):
    permutation = np.random.choice(weights.shape[0], c, replace=False)
    hash_code = np.argmax(weights[permutation, :], axis=0)
    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_code, np.arange(len(hash_code))):
        hash_dict[k].append(v)
    # make the dictionary contain numpy arrays and not a list (for faster slicing)
    for key in hash_dict:
        hash_dict[key] = np.fromiter(hash_dict[key], dtype=np.int)
    return permutation, hash_dict


def dwta(weights, c):
    permutation = np.random.choice(weights.shape[0], c, replace=False)
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
            constant[empty_bins] += 2*c
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

