import torch
import numpy as np
import os
torch.set_default_dtype(torch.float32)


def compute_accuracy_lsh(y_pred, y_true):
    top_idx = torch.argmax(y_pred, dim=1)
    bs = top_idx.size()[0]
    correct = 0
    for i in range(bs):
        if y_true[i, top_idx[i]] > 0:
            correct += 1
    return correct/bs


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
    def __init__(self, folderName, name, size, rank, hash_type):
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
        self.saveFolderName = folderName + '/' + name + '-' + hash_type + '-' + str(size) + 'workers'
        if rank == 0 and not os.path.isdir(self.saveFolderName):
            os.mkdir(self.saveFolderName)

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

        # with open(self.saveFolderName + '/ExpDescription', 'w') as f:
        #    f.write(str(self.args) + '\n')
        #    f.write(self.args.description + '\n')


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