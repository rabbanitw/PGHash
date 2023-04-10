import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable
from lsh import pg_hashtable
from misc import compute_accuracy_lsh
import time
torch.set_default_dtype(torch.float32)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class LSHLayer(nn.Module):
    def __init__(self, layer_size, K, L, num_class, sdim, cr, lsh_freq):
        super(LSHLayer, self).__init__()
        self.layer_size = layer_size
        self.K = K
        self.L = L
        self.sdim = sdim
        self.num_class = num_class
        self.lsh_freq = lsh_freq
        self.store_query = True
        self.num_c_layers = int(cr * num_class)
        # last layer
        # add an extra dummy layer for padding
        # self.params = nn.Linear(layer_size, num_class+1)
        # self.params.bias = nn.Parameter(torch.Tensor(num_class+1, 1))

        self.params = nn.Linear(layer_size, num_class)
        self.params.bias = nn.Parameter(torch.Tensor(num_class, 1))

        initrange = 0.05
        self.params.weight.data.uniform_(-initrange, initrange)
        self.params.bias.data.fill_(0)
        # initialize LSH
        self.hash_info = [[] for _ in range(self.L)]
        self.hash_dicts = [[] for _ in range(self.L)]
        self.iteration = 0
        # self.rehash()

    def rehash(self):
        weights = self.params.weight.detach().numpy().T
        weights = weights[:, :-1]
        for i in range(self.L):
            gaussian, hash_dict = pg_hashtable(weights, self.layer_size, self.K, self.sdim)
            self.hash_info[i] = gaussian
            self.hash_dicts[i] = hash_dict

    def lsh_vanilla(self, input, label):

        bs = input.shape[1]
        bs_range = np.arange(bs)
        local_active_counter = np.zeros((bs, self.num_class), dtype=bool)
        global_active_counter = np.zeros(self.num_class, dtype=bool)
        full_size = np.arange(self.num_class)
        prev_global = None
        unique = 0

        for i in range(self.L):
            # load gaussian matrix
            gaussian = self.hash_info[i]
            hash_dict = self.hash_dicts[i]

            # Apply PG to input vector.
            transformed_layer = np.heaviside(gaussian @ input, 0)

            # convert  data to base 2 to remove repeats
            base2_hash = transformed_layer.T.dot(1 << np.arange(transformed_layer.T.shape[-1]))
            for j in bs_range:
                hc = base2_hash[j]
                active_neurons = hash_dict[hc]
                local_active_counter[j, active_neurons] = True
                global_active_counter[active_neurons] = True
            unique = np.count_nonzero(global_active_counter)
            if unique >= self.num_c_layers:
                break
            else:
                prev_global = np.copy(global_active_counter)

        # remove selected neurons (in a smart way)
        print(unique)
        gap = unique - self.num_c_layers
        if gap > 0:
            if prev_global is None:
                p = global_active_counter / unique
            else:
                # remove most recent selected neurons if multiple tables are used
                change = global_active_counter != prev_global
                p = change / np.count_nonzero(change)

            deactivate = np.random.choice(full_size, size=gap, replace=False, p=p)
            global_active_counter[deactivate] = False

            neurons_per_sample = []
            max_neurons = 0
            for k in bs_range:
                # shave off deactivated neurons
                lac = local_active_counter[k, :] * global_active_counter
                neurons = np.count_nonzero(lac)
                neurons_per_sample.append(neurons)
                # select only active neurons for this sample
                # local_active_counter[k] = full_size[lac]
                if neurons > max_neurons:
                    max_neurons = neurons

        else:
            neurons_per_sample = []
            max_neurons = 0
            for k in bs_range:
                # select only active neurons for this sample
                lac = local_active_counter[k, :]
                neurons = np.count_nonzero(lac)
                neurons_per_sample.append(neurons)
                # local_active_counter[k] = full_size[lac]
                # local_active_counter[k, :] = full_size[lac]
                if neurons > max_neurons:
                    max_neurons = neurons

        '''
        padded_active_neurons = self.num_class * np.ones((bs, max_neurons))
        # new_label = np.zeros((bs, max_neurons))
        for k in bs_range:
            lac = local_active_counter[k]
            nps = neurons_per_sample[k]
            padded_active_neurons[k, :nps] = lac
            # new_label[k, :nps] = label[k, lac]
        '''

        # make probability distribution
        non_zeros = np.count_nonzero(label, axis=1, keepdims=True)
        new_label = label[:, global_active_counter]
        new_label = new_label / non_zeros

        local_active_counter = np.delete(local_active_counter, np.logical_not(global_active_counter), axis=1)

        global_active_counter = full_size[global_active_counter]

        # return padded_active_neurons, new_label, global_active_counter, neurons_per_sample
        return local_active_counter, new_label, global_active_counter, neurons_per_sample

    def train_forward(self, x, y, lsh=True):

        if lsh:
            if self.iteration % self.lsh_freq == 0:
                self.rehash()
            self.iteration += 1

            # sample activated neuron (SAN), global activated neuron (GAN)
            dense_data = x.to_dense().t()
            dense_label = y.to_dense()
            san, target, gan, nps = self.lsh_vanilla(dense_data.data.cpu().numpy(), dense_label.data.cpu().numpy())

            # idea: add an extra weight to weight matrix (and bias to bias) that is all 0 and use that as the padding/padding idx
            new_targets = Variable(torch.from_numpy(target)).to(device)
            # sample_ids = Variable(torch.from_numpy(np.asarray(san, dtype=np.int64)), requires_grad=False).to(device)
            sample_ids = Variable(torch.from_numpy(gan), requires_grad=False).to(device)
            # sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True, padding_idx=self.num_class)
            sample_weights = F.embedding(sample_ids, self.params.weight, sparse=True)
            sample_bias = self.params.bias.squeeze()[sample_ids]

            logits = torch.matmul(x, sample_weights.t()) + sample_bias

            logits[np.logical_not(san)] = torch.finfo(torch.float32).min
            log_sm = F.log_softmax(logits, dim=1)
            log_sm = torch.mul(log_sm, torch.from_numpy(san))

            acc1 = compute_accuracy_lsh(logits, target)
            print(acc1)

            return -torch.mean(torch.sum(log_sm * new_targets, dim=1))

            '''
            # compute loss
            bs = sample_weights.size(dim=0)
            num_output = sample_weights.size(dim=1)
            logits = torch.empty((bs, num_output))
            for i in range(bs):
                w = torch.squeeze(sample_weights[i, :, :])
                data = dense_data[:, i]
                logits[i, :] = w.matmul(data) + torch.squeeze(sample_bias[i, :])
                logits[i, nps[i]:] = torch.finfo(torch.float32).min
            loss = -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * new_targets, dim=1))
            

            # ensure padded bias term stays at zero
            # add second bias term to correct before performing log-softmax

            return loss
            '''

        else:
            y = y.to_dense()
            y = y / np.count_nonzero(y, axis=1, keepdims=True)
            logits = torch.matmul(x, self.params.weight.t()) + self.params.bias.squeeze()
            acc1 = compute_accuracy_lsh(logits, y)
            print(acc1)
            return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * y, dim=1))

    def forward(self, x, y):
        if self.training:
            return self.train_forward(x, y)
        else:
            return torch.matmul(x, self.params.weight.t()) + self.params.bias.squeeze()

class Net(nn.Module):
    def __init__(self, input_size, layer_size, output_size, K, L, sdim, cr, lsh_freq):
        super(Net, self).__init__()
        stdv = 1. / np.sqrt(input_size)
        self.input_size = input_size
        self.output_size = output_size
        self.layer_size = layer_size
        self.fc = nn.Embedding(self.input_size + 1, layer_size, padding_idx=input_size, sparse=True)
        self.bias = nn.Parameter(torch.Tensor(layer_size))
        self.bias.data.uniform_(-stdv, stdv)

        self.linear = nn.Linear(input_size, layer_size)
        self.linear2 = nn.Linear(layer_size, output_size)

        self.lshLayer = LSHLayer(layer_size, K, L, output_size, sdim, cr, lsh_freq)

    def forward(self, x, y):
        # emb = torch.sum(self.fc(x), dim=1)
        # emb = emb / torch.norm(emb, dim=1, keepdim=True)
        # input = F.relu(emb + self.bias)
        input = self.linear(x)
        return self.lshLayer.forward(input, y)

        # logits = self.linear2(input).to_dense()
        # y = y.to_dense()
        # y = y / np.count_nonzero(y, axis=1, keepdims=True)
        # return -torch.mean(torch.sum(F.log_softmax(logits, dim=1) * y, dim=1))
