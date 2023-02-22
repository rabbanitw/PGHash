import tensorflow as tf
import numpy as np
import os
import shutil
from itertools import product


def compute_accuracy_lsh(y_pred, y_true, lsh_idx, num_l, topk=1):
    # result_idx = find_topk(y_pred.numpy(), topk)
    val, result_idx = tf.math.top_k(y_pred, k=topk)
    batches = y_pred.get_shape()[0]
    true_idx = y_true.indices.numpy()
    true_idx_vals = true_idx[:, 0]*num_l + true_idx[:, 1]
    count = 0
    for i in range(topk):
        pre_transform_y = result_idx[0:batches, i]
        pred_idx_vals = lsh_idx[pre_transform_y] + np.arange(batches) * num_l
        count += len(np.intersect1d(true_idx_vals, pred_idx_vals, assume_unique=True))
    return count/(batches*topk)


def find_topk(input, k, axis=1, ascending=False):
    if not ascending:
        input *= -1
    ind = np.argpartition(input, k, axis=axis)
    ind = np.take(ind, np.arange(k), axis=axis)  # k non-sorted indices
    input = np.take_along_axis(input, ind, axis=axis)  # k non-sorted values
    # sort within k elements
    ind_part = np.argsort(input, axis=axis)
    ind = np.take_along_axis(ind, ind_part, axis=axis)
    return ind


def get_ham_dist_dict(k):
    x = [i for i in product(range(2), repeat=k)]
    x = np.array(x).T
    base2_hash = x.T.dot(1 << np.arange(x.T.shape[-1]))
    outer_dict = {}
    for hash_num in range(x.shape[1]):
        hash = x[:, hash_num, np.newaxis]
        hamm_dists = np.count_nonzero(x != hash, axis=0).astype(np.int)
        inner_dict = {}
        for i in range(len(hamm_dists)):
            inner_dict[base2_hash[i]] = hamm_dists[i]
        outer_dict[base2_hash[hash_num]] = inner_dict
    return outer_dict


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Recorder(object):
    def __init__(self, folderName, size, rank, args):
        self.record_epoch_times = list()
        self.record_comp_times = list()
        self.record_comm_times = list()
        self.record_lsh_times = list()
        self.record_losses = list()
        self.record_training_acc1 = list()
        self.record_test_acc1 = list()
        self.record_avg_losses = list()
        self.record_avg_training_acc1 = list()
        self.rank = rank
        self.saveFolderName = folderName + '/' + args.name + '-' + args.hash_type + '-' + args.dataset + '-' \
                              + str(size) + 'workers-' + str(args.cr) + 'cr'
        if rank == 0:
            if not os.path.isdir(self.saveFolderName):
                os.mkdir(self.saveFolderName)
                with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                    f.write(str(args) + '\n')
            else:
                shutil.rmtree(self.saveFolderName)
                os.mkdir(self.saveFolderName)
                with open(self.saveFolderName + '/ExpDescription', 'w') as f:
                    f.write(str(args) + '\n')

    def get_saveFolder(self):
        return self.saveFolderName

    def add_new(self, epoch_time, comp_time, comm_time, lsh_time, train_acc1, test_acc1, losses,
                avg_acc1, avg_losses):
        self.record_epoch_times.append(epoch_time)
        self.record_comp_times.append(comp_time)
        self.record_comm_times.append(comm_time)
        self.record_lsh_times.append(lsh_time)
        self.record_training_acc1.append(train_acc1)
        self.record_test_acc1.append(test_acc1)
        self.record_losses.append(losses)
        self.record_avg_training_acc1.append(avg_acc1)
        self.record_avg_losses.append(avg_losses)

    def add_testacc(self, test_acc):
        self.record_test_acc1.append(test_acc)

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.record_epoch_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comp-time.log', self.record_comp_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comm-time.log', self.record_comm_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-lsh-time.log', self.record_lsh_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-loss.log', self.record_losses, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-avg-epoch-train-loss.log', self.record_avg_losses,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc-top1.log', self.record_training_acc1,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-avg-epoch-train-acc-top1.log',
                   self.record_avg_training_acc1, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc-top1.log', self.record_test_acc1,
                   delimiter=',')
