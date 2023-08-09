import numpy as np
from network import SparseNN
from mpi4py import MPI


class ModelHub:
    """
    This class performs all the vital dynamic model updates for SLIDE and PGHash to work in TF.
    """

    def __init__(self, device, num_labels, num_features, hidden_layer_size, c, k, cr, rank, size, influence, i1=0, i2=1,
                 sampled_softmax=0, ss_frac=0.1):
        """
        Initializes the base model hub class.
        :param num_labels: Dimensionality of the labels
        :param num_features: Dimensionality of the features
        :param hidden_layer_size: Size of the one hidden layer in the MLP
        :param c: Sketch (compression) dimension
        :param k: Hash length
        :param cr: Compression ratio
        :param rank: Rank of a given process (process ID)
        :param size: Total number of processes
        :param influence: The weighting that each process carries during averaging (default is uniform)
        :param i1: Periodic averaging parameter, monitors how many consecutive local updates occur
        :param i2: Periodic averaging parameter, monitors how many consecutive averaging steps occur
        :param sampled_softmax: Boolean which signals if the Sampled Softmax algorithm is used
        :param ss_frac: The fraction of neurons sampled by Sampled Softmax
        """

        # initialize all parameters
        self.nl = num_labels
        self.nf = num_features
        self.hls = hidden_layer_size
        self.c = c
        self.k = k
        self.cr = cr
        self.rank = rank
        self.size = size
        self.influence = influence
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        self.comm_iter = 0
        self.used_idx = np.zeros(self.nl)
        self.ci = np.arange(self.nl)
        self.final_dense = None
        self.full_layer_shapes = None
        self.full_layer_sizes = None
        self.device_idxs = None
        self.unique = None
        self.unique_len = None
        self.unique_idx = None
        self.count = None
        self.big_model = None
        self.full_size = np.arange(self.nl)

        # initialize the start of the compressed network weights and the total number of compressed labels
        self.num_c_layers = int(self.cr * self.nl)
        self.weight_idx = (self.nf * self.hls) + self.hls

        # use sampled softmax network if enabled
        if not sampled_softmax:
            self.model = SparseNN(self.nf, self.hls, self.nl)
            self.model = self.model.to(device)

        else:
            exit()
