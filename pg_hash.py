import numpy as np
import tensorflow as tf
import time
from lsh import pg, slide, pg_hashtable, slide_hashtable
from mlp import SparseNeuralNetwork
from misc import compute_accuracy_lsh
from mpi4py import MPI
import time


class ModelHub:

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type,
                 rank, size, q, influence, i1, i2):

        # initialize all parameters
        self.nl = num_labels
        self.nf = num_features
        self.hls = hidden_layer_size
        self.sdim = sdim
        self.num_tables = num_tables
        self.cr = cr
        self.q = q
        self.hash_type = hash_type
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

        # initialize the start of the compressed network weights and the total number of compressed labels
        self.num_c_layers = int(self.cr * self.nl)
        self.weight_idx = (self.nf * self.hls) + self.hls

        # initialize model
        worker_layer_dims = [self.nf, self.hls, self.num_c_layers]
        self.model = SparseNeuralNetwork(worker_layer_dims)
        self.layer_shapes, self.layer_sizes = self.get_model_architecture()

        if self.cr < 1:
            initializer = tf.keras.initializers.GlorotUniform()
            initial_final_dense = initializer(shape=(self.hls, self.nl)).numpy()
            partial_model = self.flatten_weights(self.model.get_weights())
            partial_size = partial_model.size
            missing_bias = self.nl - self.num_c_layers
            missing_weights = self.hls * (self.nl - self.num_c_layers)
            full_dense_weights = self.hls * self.nl
            self.full_model = np.zeros(partial_size + missing_weights + missing_bias)
            self.full_model[:self.weight_idx] = partial_model[:self.weight_idx]
            self.full_model[self.weight_idx:(self.weight_idx + full_dense_weights)] = initial_final_dense.T.flatten()
        elif self.cr > 1:
            print('ERROR: Compression Ratio is Greater Than 1 Which is Impossible!')
        else:
            print('No Compression Being Used')
            self.full_model = self.flatten_weights(self.model.get_weights())

        # determine where the bias index starts
        self.bias_start = self.full_model.size - self.nl
        self.bias_idx = self.ci + self.bias_start
        self.dense_shape = (self.hls, self.nl)

        # make all models start at the same initial model
        recv_buffer = np.empty_like(self.full_model)
        MPI.COMM_WORLD.Allreduce((1/self.size) * self.full_model, recv_buffer, op=MPI.SUM)
        self.full_model = recv_buffer
        self.update_model()

        # get back to smaller size
        # self.ci = np.arange(self.num_c_layers)
        self.ci = np.sort(np.random.choice(self.nl, self.num_c_layers, replace=False))
        self.bias_idx = self.ci + self.bias_start

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

        # wait for all workers
        MPI.COMM_WORLD.Barrier()

    def get_model_architecture(self):
        model_weights = self.model.get_weights()
        layer_shapes = []
        layer_sizes = []
        for i in range(len(model_weights)):
            layer_shapes.append(model_weights[i].shape)
            layer_sizes.append(model_weights[i].size)
        return layer_shapes, layer_sizes

    def get_final_dense(self):
        self.final_dense = self.full_model[self.weight_idx:self.bias_start].reshape(self.dense_shape)

    def get_partial_model(self):
        return self.unflatten_weights(self.full_model[:self.weight_idx])

    def flatten_weights(self, weight_list):
        return np.concatenate(list(weight_list[i].flatten() for i in range(len(weight_list))))

    def unflatten_weights(self, flat_weights):
        unflatten_model = []
        start_idx = 0
        end_idx = 0
        for i in range(len(self.layer_shapes)):
            layer_size = self.layer_sizes[i]
            end_idx += layer_size
            unflatten_model.append(flat_weights[start_idx:end_idx].reshape(self.layer_shapes[i]))
            start_idx += layer_size
        return unflatten_model

    def unflatten_weights_big(self, flat_weights):
        unflatten_model = []
        start_idx = 0
        end_idx = 0
        for i in range(len(self.full_layer_shapes)):
            layer_size = self.full_layer_sizes[i]
            end_idx += layer_size
            unflatten_model.append(flat_weights[start_idx:end_idx].reshape(self.full_layer_shapes[i]))
            start_idx += layer_size
        return unflatten_model

    def update_full_model(self, model):
        # update full model before averaging
        weights = model.get_weights()

        w = weights[-2]
        b = weights[-1]
        self.get_final_dense()
        self.final_dense[:, self.ci] = w
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()
        self.full_model[self.bias_idx] = b
        # update the first part of the model as well!
        partial_model = self.flatten_weights(weights[:-2])
        self.full_model[:self.weight_idx] = partial_model

    def update_model(self, return_model=False):

        # get biases
        biases = self.full_model[self.bias_idx]
        # get weights
        self.get_final_dense()
        weights = self.final_dense[:, self.ci]
        # set new sub-model
        sub_model = np.concatenate((self.full_model[:self.weight_idx], weights.flatten(), biases.flatten()))

        new_weights = self.unflatten_weights(sub_model)
        self.model.set_weights(new_weights)
        if return_model:
            return self.model

    def test_full_model(self, test_data, acc_meter):
        self.big_model.set_weights(self.unflatten_weights_big(self.full_model))
        label_idx = np.arange(self.nl)
        for (x_batch_test, y_batch_test) in test_data:
            test_batch = x_batch_test.get_shape()[0]
            y_pred_test = self.big_model(x_batch_test, training=False)
            test_acc1 = compute_accuracy_lsh(y_pred_test, y_batch_test, label_idx, self.nl)
            acc_meter.update(test_acc1, test_batch)
        return acc_meter.avg


class PGHash(ModelHub):

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type,
                 rank, size, q, influence, i1, i2):

        super().__init__(num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type,
                 rank, size, q, influence, i1, i2)

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

    def lsh_hamming(self, model, data, num_random_table=1):

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
            # create gaussian matrix
            pg_gaussian = (1 / int(self.hls / self.sdim)) * np.tile(np.random.normal(size=(self.sdim, self.sdim)),
                                                                    int(np.ceil(self.hls / self.sdim)))[:, :self.hls]

            # Apply PGHash to weights.
            hash_table = np.heaviside(pg_gaussian @ self.final_dense, 0)

            # Apply PG to input vector.
            transformed_layer = np.heaviside(pg_gaussian @ in_layer.T, 0)
            for j in range(bs):

                # compute hamming distances
                # diff = hash_table - transformed_layer[:, j, np.newaxis]
                # hamm_dists = np.count_nonzero(diff == 0, axis=0)/self.sdim

                hamm_dists = np.count_nonzero(hash_table != transformed_layer[:, j, np.newaxis], axis=0)

                # create list of average hamming distances for a single neuron
                if i == 0:
                    cur_idx[j] = hamm_dists
                elif i < num_random_table - 1:
                    cur_idx[j] += hamm_dists

                # compute the topk closest average hamming distances to neuron
                if i == num_random_table - 1:
                    cur_idx[j] = np.argsort(cur_idx[j])[:self.num_c_layers]

        chosen_idx = np.unique(np.concatenate(cur_idx))
        if len(chosen_idx) > self.num_c_layers:
            chosen_idx = np.sort(np.random.choice(chosen_idx, self.num_c_layers, replace=False))

        for j in range(bs):
            cur_idx[j] = np.intersect1d(chosen_idx, cur_idx[j]).astype(np.int32)

        self.ci = chosen_idx

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, cur_idx

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
                    #'''
                    if j == bs-1:
                        cur_idx[j] = np.union1d(cur_idx[j], random_fill[j * per_batch:]).astype(
                            np.int32)
                    else:
                        cur_idx[j] = np.union1d(cur_idx[j], random_fill[j*per_batch:(j+1)*per_batch]).astype(np.int32)
                    #'''
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

        self.ci = chosen_idx

        # update indices with new current index
        self.bias_idx = self.ci + self.bias_start

        return self.ci, cur_idx

    def exchange_idx(self):
        if self.rank == 0:
            self.device_idxs = np.empty((self.size, len(self.ci)), dtype=np.int32)
        MPI.COMM_WORLD.Gather(self.ci, self.device_idxs, root=0)
        if self.rank == 0:
            self.unique, self.count = np.unique(self.device_idxs.flatten(), return_counts=True)
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
            updated_final_layer[:, self.unique_idx[self.device_idxs[0, :]]] += send_final_layer
            updated_final_bias[self.unique_idx[self.device_idxs[0, :]]] += send_final_bias
            t = time.time()
            # for memory I didnt gather
            for device in range(1, self.size):
                recv_buffer_layer = np.empty(self.hls * self.num_c_layers)
                recv_buffer_bias = np.empty(self.num_c_layers)
                # receive and update final layer
                MPI.COMM_WORLD.Recv(recv_buffer_layer, source=device)
                updated_final_layer[:, self.unique_idx[self.device_idxs[device, :]]] += recv_buffer_layer.reshape(self.hls, self.num_c_layers)
                # receive and update final bias
                MPI.COMM_WORLD.Recv(recv_buffer_bias, source=device)
                updated_final_bias[self.unique_idx[self.device_idxs[device, :]]] += recv_buffer_bias
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


class SLIDE(ModelHub):

    def __init__(self, num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type,
                 rank, size, q, influence, i1, i2):

        self.gaussian_mats = None
        self.hash_dicts = []

        super().__init__(num_labels, num_features, hidden_layer_size, sdim, num_tables, cr, hash_type,
                         rank, size, q, influence, i1, i2)

    def lsh_get_hash(self):

        # get weights
        self.get_final_dense()
        n = self.final_dense.shape[0]

        gaussian_mats = None

        # determine all the hash tables and gaussian matrices
        for i in range(self.num_tables):
            g_mat, ht_dict = slide_hashtable(self.final_dense, n, self.sdim)

            if i == 0:
                gaussian_mats = g_mat
            else:
                gaussian_mats = np.vstack((gaussian_mats, g_mat))

            self.hash_dicts.append(ht_dict)

        self.gaussian_mats = gaussian_mats

    def lsh(self, data, union=True, num_random_table=50):

        # get input layer for LSH
        feature_extractor = tf.keras.Model(
            inputs=self.model.inputs,
            outputs=self.model.layers[2].output,  # this is the post relu
            # outputs=self.model.layers[1].output,  # this is the pre relu
        )

        in_layer = feature_extractor(data).numpy()
        bs = in_layer.shape[0]
        cur_idx = [i for i in range(bs)]

        if union:
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

        else:
            prev_cur_idx = [i for i in range(bs)]
            gap_idx = -np.ones(bs, dtype=np.int64)

            for i in range(self.num_tables):
                cur_gauss = self.gaussian_mats[(i*self.sdim):((i+1)*self.sdim), :]
                cur_ht_dict = self.hash_dicts[i]
                hash_idxs = slide(in_layer, cur_gauss, cur_ht_dict)
                for j in range(bs):

                    # if already filled, then skip
                    if gap_idx[j] == 0:
                        continue

                    if i == 0:
                        cur_idx[j] = hash_idxs[j]
                    else:
                        cur_idx[j] = np.intersect1d(cur_idx[j], hash_idxs[j])
                    gap_idx[j] = int(self.num_c_layers - len(cur_idx[j]))

                    # if we have not filled enough, then randomly select indices from the previous cur_idx to fill the gap
                    if gap_idx[j] > 0:
                        if i == 0:
                            prev_dropped_idx = np.setdiff1d(np.arange(self.nl), cur_idx[j])
                        else:
                            prev_dropped_idx = np.setdiff1d(prev_cur_idx[j], cur_idx[j])
                        cur_idx[j] = np.union1d(cur_idx[j], np.random.choice(prev_dropped_idx, gap_idx[j], replace=False))
                        gap_idx[j] = 0
                        continue

                    prev_cur_idx[j] = cur_idx[j]

                    # if X tables is not enough, take a random choice of the leftover (very unlikely)
                    if i == self.num_tables - 1 and gap_idx[j] < 0:
                        cur_idx[j] = np.random.choice(cur_idx[j], self.num_c_layers)

                # if all(gap_idx == 0):
                if all(gap_idx > 0):
                    break

        return cur_idx

    def update(self, prev_weights, non_active_indices):
        # update full model before averaging
        w = prev_weights[-2]
        b = prev_weights[-1]
        self.get_final_dense()
        self.final_dense[:, non_active_indices] = w[:, non_active_indices]
        self.full_model[self.weight_idx:self.bias_start] = self.final_dense.flatten()
        self.full_model[non_active_indices + self.bias_start] = b[non_active_indices]

        # update the first part of the model as well!
        # partial_model = self.flatten_weights(prev_weights[:-2])
        # self.full_model[:self.weight_idx] = partial_model

        # set new weights
        new_weights = self.unflatten_weights(self.full_model)
        self.model.set_weights(new_weights)

        return self.model
