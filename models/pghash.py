import numpy as np
import tensorflow as tf
from mpi4py import MPI
from models.base import ModelHub
from util.mlp import SparseNeuralNetwork
from util.misc import compute_accuracy_lsh
import time


class PGHash(ModelHub):

    def __init__(self, num_labels, num_features, rank, size, influence, args):

        super().__init__(num_labels, num_features, args.hidden_layer_size, args.sdim, args.num_tables, args.cr, rank,
                         size, args.q, influence)

        # initialize full model for the root device for testing accuracy
        if self.rank == 0:
            self.big_model = SparseNeuralNetwork([self.nf, self.hls, self.nl])
            mw = self.big_model.get_weights()
            layer_shapes = []
            layer_sizes = []
            for i in range(len(mw)):
                layer_shapes.append(mw[i].shape)
                layer_sizes.append(mw[i].size)
            self.full_layer_sizes = layer_sizes
            self.full_layer_shapes = layer_shapes

            # set big model weights
            self.big_model.set_weights(self.unflatten_weights_big(self.full_model))

    def test_full_model(self, test_data, acc_meter):
        self.big_model.set_weights(self.unflatten_weights_big(self.full_model))
        label_idx = np.arange(self.nl)
        for (x_batch_test, y_batch_test) in test_data:
            test_batch = x_batch_test.get_shape()[0]
            y_pred_test = self.big_model(x_batch_test, training=False)
            test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
            acc_meter.update(test_acc1, test_batch)
        return acc_meter.avg

    def lsh_hamming(self, model, data):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        cur_idx = [np.empty((self.num_tables, self.nl)) for _ in range(bs)]

        for i in range(self.num_tables):
            # create gaussian matrix
            pg_gaussian = (1 / int(self.hls / self.sdim)) * np.tile(np.random.normal(size=(self.sdim, self.sdim)),
                                                                    int(np.ceil(self.hls / self.sdim)))[:, :self.hls]

            # Apply PGHash to weights.
            hash_table = np.heaviside(pg_gaussian @ self.final_dense, 0)

            # Apply PG to input vector.
            transformed_layer = np.heaviside(pg_gaussian @ in_layer.T, 0)

            # convert  data to base 2 to remove repeats
            base2_hash = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))
            for j in range(bs):

                if base2_hash[j] in base2_hash[:j]:
                    # if hamming distance is already computed
                    h_idx = np.where(base2_hash[:j] == base2_hash[j])
                    h_idx = h_idx[0][0]
                    if i == self.num_tables - 1:
                        cur_idx[j] = cur_idx[h_idx]
                    else:
                        cur_idx[j][i, :] = cur_idx[h_idx][i, :]

                else:
                    # compute hamming distances
                    hamm_dists = np.count_nonzero(hash_table != transformed_layer[:, j, np.newaxis], axis=0)

                    # create list of average hamming distances for a single neuron
                    cur_idx[j][i, :] = hamm_dists

                    # compute the topk closest average hamming distances to neuron
                    if i == self.num_tables - 1:
                        if self.num_tables > 1:
                            cur_idx[j] = np.sum(cur_idx[j], axis=0)
                            cur_idx[j] = np.argsort(cur_idx[j])[:self.num_c_layers]
                        else:
                            cur_idx[j] = np.argsort(cur_idx[j].flatten())[:self.num_c_layers]

        chosen_idx = np.unique(np.concatenate(cur_idx))

        if len(chosen_idx) > self.num_c_layers:
            chosen_idx = np.sort(np.random.choice(chosen_idx, self.num_c_layers, replace=False))

        for j in range(bs):
            cur_idx[j] = np.intersect1d(chosen_idx, cur_idx[j])

        self.ci = chosen_idx

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, cur_idx

    def exchange_idx(self):
        if self.rank == 0:
            self.device_idxs = np.empty((self.size, self.num_c_layers), dtype=np.int32)
        send_buf = -1*np.ones(self.num_c_layers, dtype=np.int32)
        send_buf[:len(self.ci)] = self.ci
        MPI.COMM_WORLD.Gather(send_buf, self.device_idxs, root=0)
        if self.rank == 0:
            temp = []
            for dev in range(self.size):
                dev_idx = self.device_idxs[dev, :]
                temp.append(dev_idx[dev_idx != -1])
            self.device_idxs = temp
            self.unique, self.count = np.unique(np.concatenate(temp, dtype=np.int32), return_counts=True)
            self.unique_len = len(self.unique)
            self.unique_idx = np.empty(self.nl, dtype=np.int64)
            self.unique_idx[self.unique] = np.arange(self.unique_len)
            MPI.COMM_WORLD.Bcast(np.array([self.unique_len]), root=0)
            MPI.COMM_WORLD.Bcast(self.unique, root=0)
        else:
            data = np.empty(1, dtype=np.int64)
            MPI.COMM_WORLD.Bcast(data, root=0)
            self.unique_len = data[0]
            data = np.empty(self.unique_len, dtype=np.int32)
            MPI.COMM_WORLD.Bcast(data, root=0)
            self.unique = data

    def smart_average_vanilla(self, model):

        # update the model
        self.update_full_model(model)
        comm_time = 0

        # create receiving buffer for first dense layer
        recv_first_layer = np.empty_like(self.full_model[:self.weight_idx])
        # Allreduce first layer of the network
        t = time.time()
        MPI.COMM_WORLD.Allreduce(self.influence * self.full_model[:self.weight_idx], recv_first_layer, op=MPI.SUM)
        comm_time += (time.time() - t)
        # update first layer
        self.full_model[:self.weight_idx] = recv_first_layer

        # prepare the layers and biases to send
        send_final_layer = self.final_dense[:, self.ci]
        send_final_bias = self.full_model[self.bias_idx]

        if self.rank == 0:
            updated_final_layer = np.zeros((self.hls, self.unique_len))
            updated_final_bias = np.zeros(self.unique_len)
            updated_final_layer[:, self.unique_idx[self.device_idxs[0]]] += send_final_layer
            updated_final_bias[self.unique_idx[self.device_idxs[0]]] += send_final_bias
            t = time.time()
            # for memory I do not use mpi.gather
            for device in range(1, self.size):
                recv_buffer_layer = np.empty(self.hls * self.num_c_layers)
                recv_buffer_bias = np.empty(self.num_c_layers)
                # receive and update final layer
                MPI.COMM_WORLD.Recv(recv_buffer_layer, source=device)
                updated_final_layer[:, self.unique_idx[self.device_idxs[device]]] \
                    += recv_buffer_layer.reshape(self.hls, self.num_c_layers)
                # receive and update final bias
                MPI.COMM_WORLD.Recv(recv_buffer_bias, source=device)
                updated_final_bias[self.unique_idx[self.device_idxs[device]]] += recv_buffer_bias
            # perform uniform averaging
            updated_final_layer = updated_final_layer / self.count
            updated_final_bias = updated_final_bias / self.count
            # send updated layer back to all devices
            MPI.COMM_WORLD.Bcast(updated_final_layer, root=0)
            MPI.COMM_WORLD.Bcast(updated_final_bias, root=0)
            comm_time += (time.time() - t)
        else:
            t = time.time()
            # send sub architecture to root
            MPI.COMM_WORLD.Send(send_final_layer.flatten(), dest=0)
            MPI.COMM_WORLD.Send(send_final_bias, dest=0)
            # receive updated changed final layer weights from root
            updated_final_layer = np.empty((self.hls, self.unique_len))
            updated_final_bias = np.empty(self.unique_len)
            MPI.COMM_WORLD.Bcast(updated_final_layer, root=0)
            MPI.COMM_WORLD.Bcast(updated_final_bias, root=0)
            comm_time += (time.time() - t)

        # update the full model
        # set biases
        self.full_model[self.bias_start + self.unique] = updated_final_bias
        # set weights
        self.final_dense[:, self.unique] = updated_final_layer
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()

        # update the sub-architecture
        self.update_model()

        return self.model, comm_time

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

    def communicate(self, model, smart=True):

        # have to have this here because of the case that i1 = 0 (cant do 0 % 0)
        self.iter += 1
        comm_time = 0
        # I1: Number of Local Updates Communication Set
        if self.iter % (self.i1 + 1) == 0:
            self.comm_iter += 1
            if smart:
                model, t = self.smart_average_vanilla(model)
            else:
                model, t = self.simple_average(model)
            comm_time += t
            # I2: Number of Consecutive 1-Step Averaging
            if self.comm_iter % self.i2 == 0:
                self.comm_iter = 0
            else:
                # decrease iteration by one in order to run another one update and average step (I2 communication)
                self.iter -= 1
        return model, comm_time


'''
    def lsh_initial(self, model, data):

        # get weights
        self.get_final_dense()
        n = self.final_dense.shape[0]

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            # outputs=model.layers[2].output,  # this is the post relu
            outputs=model.layers[1].output,  # this is the pre relu
        )
        in_layer = feature_extractor(data).numpy()

        bs = in_layer.shape[0]
        ham_dists = np.zeros(self.nl)

        # run LSH to find the most important weights over the entire next Q batches
        for _ in range(self.num_tables):
            g_mat, ht_dict = pg_hashtable(self.final_dense, n, self.sdim)
            ham_dists += pg(in_layer, g_mat, ht_dict, ham_dists)

        # pick just the largest differences
        avg_ham_dists = -ham_dists / (bs * self.num_tables)

        # union the indices of each batch
        self.ci = np.sort(tf.math.top_k(avg_ham_dists, self.num_c_layers).indices.numpy())

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, [self.ci for _ in range(bs)]

    def lsh_vanilla(self, model, data, num_random_table=20):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
            # outputs=self.model.layers[1].output,  # this is the pre relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        cur_idx = [i for i in range(bs)]

        for i in range(num_random_table):
            g_mat, ht_dict = pg_hashtable(self.final_dense, self.hls, self.sdim)

            # Apply PG to input vector.
            transformed_layer = np.heaviside(g_mat @ in_layer.T, 0)
            # convert to base 2
            hash_code = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))
            for j in range(bs):
                if i == 0:
                    cur_idx[j] = ht_dict[hash_code[j]]
                else:
                    cur_idx[j] = np.union1d(cur_idx[j], ht_dict[hash_code[j]])

            chosen_idx, count = np.unique(np.concatenate(cur_idx), return_counts=True)

            if len(chosen_idx) > self.num_c_layers:
                # take the top cr*num_labels neurons
                top_idx = np.argsort(-count)[:self.num_c_layers]
                chosen_idx = chosen_idx[top_idx]
                for j in range(bs):
                    cur_idx[j] = np.intersect1d(cur_idx[j], chosen_idx).astype(np.int32)
                self.ci = np.sort(chosen_idx).astype(np.int32)
                break
            if i == num_random_table-1:
                # try random fill first
                gap = self.num_c_layers - len(chosen_idx)
                non_chosen = np.setdiff1d(np.arange(self.nl), chosen_idx)
                random_fill = np.random.choice(non_chosen, size=(gap), replace=False)
                per_batch = int(gap/bs)

                for j in range(bs):
                    #cur_idx[j] = np.union1d(cur_idx[j], random_fill).astype(np.int32)
                    if j == bs-1:
                        cur_idx[j] = np.union1d(cur_idx[j], random_fill[j * per_batch:]).astype(
                            np.int32)
                    else:
                        cur_idx[j] = np.union1d(cur_idx[j], random_fill[j*per_batch:(j+1)*per_batch]).astype(np.int32)
                self.ci = np.union1d(random_fill, chosen_idx).astype(np.int32)

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, cur_idx

    def lsh_tables(self):

        # get weights
        self.get_final_dense()
        n = self.final_dense.shape[0]

        gaussian_mats = None
        self.hash_dicts = []

        # determine all the hash tables and gaussian matrices
        for i in range(self.num_tables):
            g_mat, ht_dict = slide_hashtable(self.final_dense, n, self.sdim)
            # g_mat, ht_dict = pg_hashtable(self.final_dense, n, self.sdim)

            if i == 0:
                gaussian_mats = g_mat
            else:
                gaussian_mats = np.vstack((gaussian_mats, g_mat))

            self.hash_dicts.append(ht_dict)

        self.gaussian_mats = gaussian_mats

    def lsh(self, model, data, num_random_table=50):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=model.inputs,
            outputs=model.layers[2].output,  # this is the post relu
            # outputs=self.model.layers[1].output,  # this is the pre relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        cur_idx = [i for i in range(bs)]

        table_idx = np.random.choice(self.num_tables, num_random_table, replace=False)
        for i in range(num_random_table):
            idx = table_idx[i]
            cur_gauss = self.gaussian_mats[(idx * self.sdim):((idx + 1) * self.sdim), :]
            cur_ht_dict = self.hash_dicts[idx]
            hash_idxs = slide(in_layer, cur_gauss, cur_ht_dict)
            for j in range(bs):
                if i == 0:
                    cur_idx[j] = hash_idxs[j]
                else:
                    cur_idx[j] = np.union1d(cur_idx[j], hash_idxs[j])

        chosen_idx = np.unique(np.concatenate(cur_idx))
        if len(chosen_idx) > self.num_c_layers:
            chosen_idx = np.sort(np.random.choice(chosen_idx, self.num_c_layers, replace=False))
            for j in range(bs):
                cur_idx[j] = np.intersect1d(chosen_idx, cur_idx[j]).astype(np.int32)
        elif len(chosen_idx) < self.num_c_layers:
            gap = self.num_c_layers - len(chosen_idx)
            per_batch = int(gap / bs)
            non_chosen = np.setdiff1d(np.arange(self.nl), chosen_idx)
            random_fill = np.random.choice(non_chosen, gap, replace=False)
            chosen_idx = np.union1d(chosen_idx, random_fill)
            for j in range(bs):
                if j == bs - 1:
                    cur_idx[j] = np.union1d(cur_idx[j], random_fill[j * per_batch:]).astype(np.int32)
                else:
                    cur_idx[j] = np.union1d(cur_idx[j], random_fill[j * per_batch:(j + 1) * per_batch]).astype(np.int32)

        self.ci = chosen_idx.astype(np.int32)

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, cur_idx
'''
