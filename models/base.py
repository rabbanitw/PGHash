import numpy as np
from util.network import SparseNeuralNetwork, SampledSoftmax
from mpi4py import MPI


class ModelHub:
    """
    This class performs all the vital dynamic model updates for SLIDE and PGHash to work in TF.
    """

    def __init__(self, num_labels, num_features, hidden_layer_size, c, k, cr, rank, size, influence, i1=0, i2=1,
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
            self.model = SparseNeuralNetwork([self.nf, self.hls, self.nl])
        else:
            self.model = SampledSoftmax([self.nf, self.hls, self.nl], int(ss_frac*self.nl))

        # create an entire vector of the flattened full model
        self.full_model = self.flatten_weights(self.model.get_weights())

        # if using a compression rate less than 1, create a local model with the correct number of final neurons
        if self.cr < 1:
            # initialize compressed model
            worker_layer_dims = [self.nf, self.hls, self.num_c_layers]
            self.model = SparseNeuralNetwork(worker_layer_dims)
            self.layer_shapes, self.layer_sizes = self.get_model_architecture()
        elif self.cr > 1:
            print('ERROR: Compression Ratio is Greater Than 1 Which is Impossible!')
            exit()
        else:
            self.layer_shapes, self.layer_sizes = self.get_model_architecture()
            print('No Compression Being Used')

        # determine where the bias index starts
        self.bias_start = self.full_model.size - self.nl
        self.bias_idx = self.ci + self.bias_start
        self.dense_shape = (self.hls, self.nl)

        # make all devices start at same initial model
        self.sync_models()

        # get back to smaller size
        self.ci = np.sort(np.random.choice(self.nl, self.num_c_layers, replace=False))
        self.bias_idx = self.ci + self.bias_start

        # set the new model
        self.update_model()

    def get_model_architecture(self):
        """
        This function returns the shapes and sizes of each layer in the network (useful for averaging).
        :return: Shapes and sizes of each layer of a given network
        """
        # output the model weights for a given model
        model_weights = self.model.get_weights()
        layer_shapes = []
        layer_sizes = []
        # loop through each layer of the model and store the shape and size
        for i in range(len(model_weights)):
            layer_shapes.append(model_weights[i].shape)
            layer_sizes.append(model_weights[i].size)
        return layer_shapes, layer_sizes

    def sync_models(self):
        """
        This function ensures that all processes start with the same model parameters.
        :return: Uniform model amongst all processes
        """
        # make all models start at the same initial model
        recv_buffer = np.empty_like(self.full_model)
        MPI.COMM_WORLD.Allreduce((1 / self.size) * self.full_model, recv_buffer, op=MPI.SUM)
        self.full_model = recv_buffer
        self.update_model()

    def get_final_dense(self):
        """
        Function outputs the final layer weights used by a process.
        :return: Final layer weights
        """
        self.final_dense = self.full_model[self.weight_idx:self.bias_start].reshape(self.dense_shape)

    def get_partial_model(self):
        """
        Function returns the first half of the network (up to the last layer).
        :return: First half of the model
        """
        return self.unflatten_weights(self.full_model[:self.weight_idx])

    def flatten_weights(self, weight_list):
        """
        Function flattens the output of model.get_weights() into a vector.
        :param weight_list: List of weights for each layer in a model
        :return: Flattened model
        """
        return np.concatenate(list(weight_list[i].flatten() for i in range(len(weight_list))))

    def unflatten_weights(self, flat_weights):
        """
        Function unflattens the flattened model vector.
        :param flat_weights: Flattened model vector
        :return: Unflattened model (for TF)
        """
        unflatten_model = []
        start_idx = 0
        end_idx = 0
        # loop through each layer and reshape the flattened vector into its true size and shape
        for i in range(len(self.layer_shapes)):
            layer_size = self.layer_sizes[i]
            end_idx += layer_size
            unflatten_model.append(flat_weights[start_idx:end_idx].reshape(self.layer_shapes[i]))
            start_idx += layer_size
        return unflatten_model

    def unflatten_weights_big(self, flat_weights):
        """
        Function unflattens the flattened model vector for the BIG model.
        :param flat_weights: Flattened model vector (of entire full model)
        :return: Unflattened full model (for TF)
        """
        unflatten_model = []
        start_idx = 0
        end_idx = 0
        # loop through each layer and reshape the flattened vector into its true size and shape
        for i in range(len(self.full_layer_shapes)):
            layer_size = self.full_layer_sizes[i]
            end_idx += layer_size
            unflatten_model.append(flat_weights[start_idx:end_idx].reshape(self.full_layer_shapes[i]))
            start_idx += layer_size
        return unflatten_model

    def update_full_model(self, model):
        """
        Function takes a model and updates the continually stored full model vector
        :param model: Current model for a given process
        :return: Updated full model vector
        """
        # update full model before averaging
        weights = model.get_weights()

        # unpack the weights and biases of the final layer
        w = weights[-2]
        b = weights[-1]

        # receive/isolate the final layer weights
        self.get_final_dense()

        # update the final layer weights and biases with the updated values
        self.final_dense[:, self.ci] = w
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()
        self.full_model[self.bias_idx] = b

        # update the first part of the model as well!
        partial_model = self.flatten_weights(weights[:-2])
        self.full_model[:self.weight_idx] = partial_model

    def update_model(self):
        """
        Function sets the model for a process with the full model vector.
        :return: Model for a given process
        """

        # get weights and biases
        biases = self.full_model[self.bias_idx]
        self.get_final_dense()
        weights = self.final_dense[:, self.ci]

        # set new sub-model
        sub_model = np.concatenate((self.full_model[:self.weight_idx], weights.flatten(), biases.flatten()))

        # unflatten full model and set its new weights
        new_weights = self.unflatten_weights(sub_model)
        self.model.set_weights(new_weights)
