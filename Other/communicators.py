import numpy as np
import time
from mpi4py import MPI
from Old.unpack import flatten_weights, unflatten_weights, update_full_model, get_sub_model


class DecentralizedSGD:
    """
    decentralized averaging according to a topology sequence
    For DSGD: Set i1 = 0 and i2 > 0 (any number it doesn't matter)
    For PD-SGD: Set i1 > 0 and i2 = 1
    For LD-SGD: Set i1 > 0 and i2 > 1
    """

    def __init__(self, rank, size, comm, topology, layer_shapes, layer_sizes, i1, i2):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.topology = topology
        self.neighbor_list = self.topology.neighbor_list
        self.neighbor_weights = topology.neighbor_weights
        self.degree = len(self.neighbor_list)
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        self.comm_iter = 0

    def prepare_comm_buffer(self, model):
        model_weights = model.weights()
        # flatten tensor weights
        self.send_buffer = flatten_weights(model_weights)
        self.recv_buffer = np.zeros_like(self.send_buffer)

    def model_sync(self, model):

        # necessary preprocess
        self.prepare_comm_buffer(model)

        # perform all-reduce to synchronize initial models across all clients
        MPI.COMM_WORLD.Allreduce(self.send_buffer, self.recv_buffer, op=MPI.SUM)
        # divide by total workers to get average model
        self.recv_buffer = self.recv_buffer / self.size

        # update local models
        self.reset_model(model)
        return unflatten_weights(self.recv_buffer, self.layer_shapes, self.layer_sizes)

    def average(self, model):

        # necessary preprocess
        self.prepare_comm_buffer(model)

        self.comm.Barrier()
        tic = time.time()

        # compute self weight according to degree
        selfweight = 1 - np.sum(self.neighbor_weights)
        # compute weighted average: (1-d*alpha)x_i + alpha * sum_j x_j
        self.recv_buffer = np.add(self.recv_buffer, selfweight*self.send_buffer)

        self.recv_tmp = np.empty_like(self.send_buffer)
        # decentralized averaging
        for idx, node in enumerate(self.neighbor_list):
            self.comm.Sendrecv(sendbuf=self.send_buffer, source=node, recvbuf=self.recv_tmp, dest=node)
            # Aggregate neighbors' models: alpha * sum_j x_j
            self.recv_buffer = np.add(self.recv_buffer, self.neighbor_weights[idx] * self.recv_tmp)

        self.comm.Barrier()
        toc = time.time()

        # update local models
        self.reset_model(model)

        return toc - tic

    def reset_model(self, model):
        # Reset local models to be the averaged model
        new_weights = unflatten_weights(self.recv_buffer, self.layer_shapes, self.layer_sizes)
        model.set_weights(new_weights)

    def communicate(self, model):

        # Have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0

        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:

            self.comm_iter += 1
            # decentralized averaging according to activated topology
            comm_time += self.average(model)

            # I2: Number of DSGD Communication Set
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1

        return comm_time


class CentralizedSGD:
    """
        centralized averaging, allowing periodic averaging
        """
    def __init__(self, rank, size, comm, influence, layer_shapes, layer_sizes, i1, i2):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.influence = influence
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        self.comm_iter = 0

    def reset_model(self, model):
        # Reset local models to be the averaged model
        new_weights = unflatten_weights(self.recv_buffer, self.layer_shapes, self.layer_sizes)
        model.set_weights(new_weights)

    def prepare_comm_buffer(self, model):
        model_weights = model.get_weights()
        # flatten tensor weights
        self.send_buffer = flatten_weights(model_weights)
        self.recv_buffer = np.zeros_like(self.send_buffer)

    def model_sync(self, model):

        # necessary preprocess
        self.prepare_comm_buffer(model)

        # perform all-reduce to synchronize initial models across all clients
        MPI.COMM_WORLD.Allreduce(self.send_buffer, self.recv_buffer, op=MPI.SUM)
        # divide by total workers to get average model
        self.recv_buffer = self.recv_buffer / self.size

        # update local models
        self.reset_model(model)
        return unflatten_weights(self.recv_buffer, self.layer_shapes, self.layer_sizes)

    def average(self, model):
        # necessary preprocess
        self.prepare_comm_buffer(model)
        # perform all-reduce to synchronize initial models across all clients
        tic = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence*self.send_buffer, self.recv_buffer, op=MPI.SUM)
        toc = time.time()
        # update local models
        self.reset_model(model)
        return toc - tic

    def communicate(self, model):
        # Have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0
        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:
            self.comm_iter += 1
            # decentralized averaging according to activated topology
            comm_time += self.average(model)
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return comm_time

class LSHCentralizedSGD:
    """
        centralized averaging, allowing periodic averaging
        """
    def __init__(self, rank, size, comm, influence, layer_shapes, layer_sizes, i1, i2, num_f, num_l, hls):
        self.comm = comm
        self.rank = rank
        self.size = size
        self.influence = influence
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        self.i1 = i1
        self.i2 = i2
        self.iter = 0
        self.comm_iter = 0
        self.start_idx_w = ((num_f * hls) + hls + (4 * hls))
        self.num_features = num_f
        self.num_labels = num_l
        self.hls = hls

    def average(self, model, full_model, cur_idx, start_idx_b):

        # update full model before averaging
        weights = model.get_weights()
        w = weights[-2]
        b = weights[-1]

        # update the first part of the model as well!

        full_model = update_full_model(full_model, w, b, cur_idx, start_idx_b, self.num_features, self.hls)

        # create receiving buffer
        recv_buffer = np.empty_like(full_model)

        # perform averaging
        tic = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence*full_model, recv_buffer, op=MPI.SUM)
        toc = time.time()

        # set new sub-model
        w, b = get_sub_model(recv_buffer, cur_idx, start_idx_b, self.num_features, self.hls)
        sub_model = np.concatenate((recv_buffer[:self.start_idx_w], w.flatten(), b.flatten()))
        new_weights = unflatten_weights(sub_model, self.layer_shapes, self.layer_sizes)
        model.set_weights(new_weights)

        return recv_buffer, toc - tic

    def communicate(self, model, full_model, cur_idx, start_idx_b, layer_shapes, layer_sizes):
        # update potentially changed layer shapes and sizes
        self.layer_shapes = layer_shapes
        self.layer_sizes = layer_sizes
        # Have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0
        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1+1) == 0:
            self.comm_iter += 1
            # decentralized averaging according to activated topology
            full_model, t = self.average(model, full_model, cur_idx, start_idx_b)
            comm_time += t
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return full_model, comm_time
