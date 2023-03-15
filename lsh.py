import numpy as np
from collections import defaultdict


def pg_hashtable(weights, n, sdim):
    '''
    compute hashing
    :param vectors:
    :param n:
    :param sdim:
    :return:
    '''

    # create gaussian matrix
    pg_gaussian = (1/int(n/sdim))*np.tile(np.random.normal(size=(sdim, sdim)), int(np.ceil(n/sdim)))[:, :n]
    # pg_gaussian = np.random.normal(size=(sdim, n))

    # Apply PGHash to weights.
    hash_table = np.heaviside(pg_gaussian@weights, 0)

    hash_table1 = hash_table[:7, :]
    #hash_table2 = hash_table[7:, :]

    # convert to base 2
    hash_table1 = hash_table1.T.dot(1 << np.arange(hash_table1.T.shape[-1]))
    # hash_table2 = hash_table2.T.dot(1 << np.arange(hash_table2.T.shape[-1]))

    # create dictionary holding the base 2 hash code (key) and the weights which share that hash code (value)
    hash_dict = defaultdict(list)
    for k, v in zip(hash_table1, np.arange(len(hash_table1))):
        hash_dict[k].append(v)

    #for k, v in zip(hash_table2, np.arange(len(hash_table2))):
    #    hash_dict[k].append(v)

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


def slide_hashtable(weights, n, sdim):

    # create gaussian matrix
    slide_gaussian = np.random.normal(size=(sdim, n))

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
