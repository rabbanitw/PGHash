import tensorflow as tf
import numpy as np
import os
from datetime import datetime


def compute_accuracy(y_true, y_pred, topk=1):
    result = tf.math.top_k(y_pred, k=topk)
    r, c = y_pred.get_shape()
    true_idx = y_true.indices.numpy()
    count = 0
    for i in range(r):
        for j in range(topk):
            top_idx = np.array([i, result.indices[i, j].numpy()])
            count += int(np.any(np.all(top_idx == true_idx, axis=1)))
    return count/(r*topk)


def compute_accuracy_lsh(y_true, y_pred, lsh_idx, topk=1):
    result = tf.math.top_k(y_pred, k=topk)
    r, c = y_pred.get_shape()
    true_idx = y_true.indices.numpy()
    result_idx = result.indices.numpy()
    count = 0
    for i in range(r):
        for j in range(topk):
            transform_idx_y = lsh_idx[result_idx[i, j]]
            top_idx = np.array([i, transform_idx_y])
            count += int(np.any(np.all(top_idx == true_idx, axis=1)))
    return count/(r*topk)


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
    def __init__(self, folderName, rank, hash_type):
        self.record_epoch_times = list()
        self.record_comp_times = list()
        self.record_comm_times = list()
        self.record_lsh_times = list()
        self.record_losses = list()
        self.record_training_acc1 = list()
        self.record_training_acc5 = list()
        self.record_avg_losses = list()
        self.record_avg_training_acc1 = list()
        self.record_avg_training_acc5 = list()
        self.rank = rank
        now = datetime.now()
        self.start_time = now.strftime("%m/%d/%Y/%H:%M")
        self.start_time = self.start_time.replace(r'/', '-')
        self.saveFolderName = folderName + '/' + hash_type + '-' + self.start_time
        if rank == 0 and not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)

    def add_new(self, epoch_time, comp_time, comm_time, lsh_time, acc1, acc5, losses,
                avg_acc1, avg_acc5, avg_losses):
        self.record_epoch_times.append(epoch_time)
        self.record_comp_times.append(comp_time)
        self.record_comm_times.append(comm_time)
        self.record_lsh_times.append(lsh_time)
        self.record_training_acc1.append(acc1)
        self.record_training_acc5.append(acc5)
        self.record_losses.append(losses)
        self.record_avg_training_acc1.append(avg_acc1)
        self.record_avg_training_acc5.append(avg_acc5)
        self.record_avg_losses.append(avg_losses)

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
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc-top5.log', self.record_training_acc5,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-avg-epoch-train-acc-top5.log',
                   self.record_avg_training_acc5, delimiter=',')

        # with open(self.saveFolderName + '/ExpDescription', 'w') as f:
        #    f.write(str(self.args) + '\n')
        #    f.write(self.args.description + '\n')
