import numpy as np
from mpi4py import MPI
from models.base import ModelHub
import time


class Dense(ModelHub):
    """
    Class performs full-training for large-scale recommender systems.
    """

    def __init__(self, num_labels, num_features, hidden_layer_size, c, k, cr, rank, size, influence,
                 sampled_softmax=0):
        """
        Initializes the full-training model.
        :param num_labels: Dimensionality of the labels
        :param num_features: Dimensionality of the features
        :param hidden_layer_size: Size of the one hidden layer in the MLP
        :param c: Sketch (compression) dimension
        :param k: Hash length
        :param cr: Compression ratio
        :param rank: Rank of a given process (process ID)
        :param size: Total number of processes
        :param influence: The weighting that each process carries during averaging (default is uniform)
        :param sampled_softmax: Boolean which signals if the Sampled Softmax algorithm is used
        """

        super().__init__(num_labels, num_features, hidden_layer_size, c, k, cr, rank, size, influence,
                         sampled_softmax=sampled_softmax)

    def simple_average(self, model):
        """
        Function that averages all weights, even those that were not updated. This function is not used in our work,
        and is only a lazy approach.
        :param model: Current model for each process
        :return: Communication time to perform the averaging
        """
        # update the full model
        self.update_full_model(model)

        # create receiving buffer
        recv_buffer = np.empty_like(self.full_model)
        # perform averaging
        tic = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence * self.full_model, recv_buffer, op=MPI.SUM)
        toc = time.time()

        # update full model
        self.full_model = recv_buffer
        # set new sub-model
        self.update_model()
        return self.model, toc - tic

    def communicate(self, model):
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
            model, t = self.simple_average(model)
            comm_time += t
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return model, comm_time
