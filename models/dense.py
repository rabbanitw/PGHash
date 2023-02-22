import numpy as np
from mpi4py import MPI
from models.base import ModelHub
import time


class Dense(ModelHub):

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, rank, size, q, influence):

        super().__init__(num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, rank, size, q, influence)

    def simple_average(self, model):

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
