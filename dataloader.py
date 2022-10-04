import numpy as np
from xclib.data import data_utils


def load_amazon670():
    print('hi')
    features, labels, num_samples, num_features, num_labels = data_utils.read_data('Data/Amazon670k/train.txt')
