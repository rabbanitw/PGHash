import torch
import numpy as np
import os
import shutil


def compute_accuracy(logits, labels_batch):
    top_k_classes = torch.argmax(logits, dim=1).cpu()
    pred = [len(np.intersect1d(top_k_classes[j], labels_batch[j])) for j in range(len(top_k_classes))]
    return np.mean(pred)


def top1acc(y_pred, y_true):
    top = torch.argmax(y_pred, dim=1)
    count = 0
    for i, idx in enumerate(top):
        if y_true[i, idx]:
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
        self.record_training_acc1 = list()
        self.record_test_acc1 = list()
        self.record_avg_losses = list()
        self.record_avg_training_acc1 = list()
        self.num_active_neurons = list()
        self.avg_active_neurons = list()
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

    def add_new(self, epoch_time, comp_time, comm_time, lsh_time, train_acc1, test_acc1, losses,
                avg_acc1, avg_losses, active_neurons, avg_neurons):
        self.record_epoch_times.append(epoch_time)
        self.record_comp_times.append(comp_time)
        self.record_comm_times.append(comm_time)
        self.record_lsh_times.append(lsh_time)
        self.record_training_acc1.append(train_acc1)
        self.record_test_acc1.append(test_acc1)
        self.record_losses.append(losses)
        self.record_avg_training_acc1.append(avg_acc1)
        self.record_avg_losses.append(avg_losses)
        self.num_active_neurons.append(active_neurons)
        self.avg_active_neurons.append(avg_neurons)

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
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-loss.log', self.record_losses,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-avg-epoch-train-loss.log', self.record_avg_losses,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-train-acc-top1.log', self.record_training_acc1,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-avg-epoch-train-acc-top1.log',
                   self.record_avg_training_acc1, delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-test-acc-top1.log', self.record_test_acc1,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-active-neurons.log', self.num_active_neurons,
                   delimiter=',')
        np.savetxt(self.saveFolderName + '/r' + str(self.rank) + '-avg-active-neurons.log', self.avg_active_neurons,
                   delimiter=',')
