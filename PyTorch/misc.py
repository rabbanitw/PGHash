import torch
import numpy as np
import os
import shutil


def top1acc(y_pred, y_true):
    top = torch.argmax(y_pred, dim=1).cpu()
    count = 0
    for i, idx in enumerate(top):
        if y_true[i][idx]:
            count += 1
    return count / len(top)


def top1acc_test(y_pred, y_true):
    top = torch.argmax(y_pred, dim=1).cpu()
    count = 0
    for i, idx in enumerate(top):
        if idx in y_true[i]:
            count += 1
    return count / len(top)


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
        self.record_train_acc = list()
        self.record_test_acc = list()
        self.epoch_test_acc = list()
        self.rank = rank
        self.saveFolderName = folderName + '/' + args.name + '-' + args.hash_type + '-' + args.dataset + '-' \
                              + str(size) + 'workers-' + str(args.cr) + 'cr-' + str(args.num_tables) + 'tables-' + \
                              str(args.steps_per_lsh) + 'rehash'

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

    def add_new(self, epoch_time, comp_time, comm_time, lsh_time, train_acc1, losses):
        self.record_epoch_times.append(epoch_time)
        self.record_comp_times.append(comp_time)
        self.record_comm_times.append(comm_time)
        self.record_lsh_times.append(lsh_time)
        self.record_train_acc.append(train_acc1)
        self.record_losses.append(losses)

    def add_test_accuracy(self, test_acc, epoch=False):
        if epoch:
            self.epoch_test_acc.append(test_acc)
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-acc-top1.log', self.epoch_test_acc,
                       delimiter=',')
        else:
            self.record_test_acc.append(test_acc)
            np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc-top1.log', self.record_test_acc,
                       delimiter=',')

    def save_to_file(self):
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-epoch-time.log', self.record_epoch_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comp-time.log', self.record_comp_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-comm-time.log', self.record_comm_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-lsh-time.log', self.record_lsh_times,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-loss.log', self.record_losses,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc-top1.log', self.record_train_acc,
                   delimiter=',')
