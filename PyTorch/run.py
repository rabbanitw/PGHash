import torch
import argparse
from dataloader import data_generator_train, data_generator_test
from misc import AverageMeter, Recorder, top1acc, top1acc_test
from pghash import PGHash
import numpy as np
import random
from network import Net
import glob
import time
from lsh import gpu_pghash_lsh, gpu_pghashd_lsh, lsh_vanilla


def rehash(model, device, SB, hash_dicts, c, k, hls, num_tables, dwta=True, slide=False):

    # Two hashing methods are PGHash and PGHash-D (DWTA variant)
    if dwta:
        # create list of possible coordinates to randomly select for DWTA
        potential_indices = np.arange(hls)
        for i in range(num_tables):
            if not slide:
                # server randomly selects coordinates for DWTA (stored as permutation list), vector is dimension c
                indices = np.random.choice(potential_indices, c, replace=False)
                # process selects a k-subset of the c coordinates to use for on-device DWTA
                perm = np.random.choice(indices, k, replace=False)
                # index the k coordinates for each neuron (size is k x n)
                perm_weight = model.fc2.weight.t()[perm, :]
            else:
                perm_weight = model.fc2.weight.t()
            # run PGHash-D LSH
            perm2, hash_dict = gpu_pghashd_lsh(device, perm_weight, k, slide=slide)
            # save the permutation list in local memory (small memory cost) and hash tables
            if not slide:
                SB[i] = perm
            else:
                SB[i] = perm2
            hash_dicts[i] = hash_dict
    else:
        for i in range(num_tables):
            # perform PGHash LSH
            # weights = self.model.linear2.weight.detach().cpu().numpy().transpose()
            weights = model.fc2.weight.t()
            # SB, hash_dict = pghash_lsh(weights, self.hls, self.k, self.c)
            gaussian, hash_dict = gpu_pghash_lsh(device, weights, hls, k, c)
            # save gaussian and hash tables
            # when rehashing is performed every step, these cn be immediately discarded
            SB[i] = gaussian
            hash_dicts[i] = hash_dict

    return SB, hash_dicts


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Delicious200K')
    parser.add_argument('--graph_type', type=str, default='fully_connected')
    parser.add_argument('--hash_type', type=str, default='pghash')
    parser.add_argument('--dwta', type=int, default=0)
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--c', type=int, default=8)
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--num_val_batches', type=int, default=100)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--cr', type=float, default=1)
    parser.add_argument('--train_bs', type=int, default=128)
    parser.add_argument('--test_bs', type=int, default=128)
    parser.add_argument('--steps_per_lsh', type=int, default=50)
    parser.add_argument('--steps_per_test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_layer_size', type=int, default=128)

    # parse the argument
    args = parser.parse_args()

    # set random seed
    randomSeed = args.randomSeed
    torch.manual_seed(randomSeed)
    np.random.seed(randomSeed)
    random.seed(randomSeed)
    torch.backends.cudnn.benchmark = False

    # determine torch device available (default to GPU if available)
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        dev = ["cuda:" + str(i) for i in range(num_gpus)]
    else:
        num_gpus = 0
        dev = ["cpu"]

    if num_gpus < 2:
        device = torch.device(dev[0])
        device2 = device
    else:
        device = torch.device(dev[0])
        device2 = torch.device(dev[1])

    # hashing parameters
    c = args.c
    k = args.k
    num_tables = args.num_tables
    hash_type = args.hash_type
    steps_per_lsh = args.steps_per_lsh
    SB = [[] for _ in range(num_tables)]
    hash_dicts = [[] for _ in range(num_tables)]

    # load base network topology
    graph_type = args.graph_type
    weight_type = None
    num_clusters = None

    # training parameters
    train_bs = args.train_bs
    test_bs = args.test_bs
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = '../data/' + args.dataset + '/train.txt'
    test_data_path = '../data/' + args.dataset + '/test.txt'
    train_files = glob.glob(train_data_path)
    test_files = glob.glob(test_data_path)

    nc = None
    nf = None
    n_train = None
    n_test = None
    dwta = None
    if args.dataset == 'Amazon670K':
        nc = 670091
        nf = 135909
        n_train = 490449
        n_test = 153025
        dwta = True

    elif args.dataset == 'Delicious200K':
        nc = 205443
        nf = 782585
        n_train = 196606
        n_test = 100095
        dwta = False

    steps_per_epoch = 196606 // args.train_bs
    n_steps = args.epochs * steps_per_epoch

    if k > c and args.dwta == 1:
        print('Error: Compression Size Smaller than Hash Length for PGHash-D')
        exit()

    if args.hash_type[:2] == 'pg':
        method = 'PGHash'
        cr = args.cr
        slide = False
        dense = False
    elif args.hash_type == 'slide':
        method = 'SLIDE'
        cr = 1
        slide = True
        dense = False
    else:
        dense = True
        cr = 1

    # load (large) dataset
    print('Loading and partitioning data...')
    # train_dl = data_generator_csr(train_data_path, train_bs, nf, nc)
    # test_dl = data_generator_csr(test_data_path, test_bs, nf, nc)

    # initialize meters
    train_acc = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('../output', 1, 0, args)

    # select method used and begin training once all devices are ready
    print('Initializing model...')
    Method = PGHash(nc, nf, 0, 1, 1, device, args, slide=slide)
    model = Net(nf, hls, nc).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load data generator
    training_data_generator = data_generator_train(train_files, train_bs, nc)
    test_data_generator = data_generator_test(test_files, test_bs)

    lsh_time = 0
    for i in range(1, n_steps+1):

        # train
        idxs_batch, vals_batch, y = next(training_data_generator)
        x = torch.sparse_coo_tensor(idxs_batch, vals_batch,
                                    size=(train_bs, nf),
                                    device=device,
                                    requires_grad=False)
        label = torch.from_numpy(y).to(device)

        if not dense:
            # compute LSH
            lsh_init = time.time()

            # rehashing step: weights of final layer are hashed and placed into buckets depending upon their hash code
            if i % steps_per_lsh == 0 or i == 1:
                SB, hash_dicts = rehash(model, device, SB, hash_dicts, c, k, hls, num_tables, dwta=dwta, slide=slide)

            # active neuron selection step: each sample in batch is hashed and the resulting hash code is used
            # to select which neurons will be activated (exact matches -- vanilla style)
            active_idx, sample_active_idx, true_neurons_bool = lsh_vanilla(model, device, x, k, num_tables, SB,
                                                                           hash_dicts, nc, dwta=dwta)

            lsh_time = time.time() - lsh_init

            # create a mask to ensure only the active neurons for each sample are used & updated during training
            # note: try to save computational time below by creating mask from indices sans for loop
            mask = np.zeros((train_bs, nc))
            for j in range(train_bs):
                mask[j, sample_active_idx[j]] = 1

            active_mask = torch.from_numpy(mask).to(device)
            softmax_mask = torch.where(active_mask == 1, 0., torch.tensor(-1e20)).to(device)


        optimizer.zero_grad()

        init_time = time.time()
        logits = model(x)


        # PGHash loss

        # dense loss
        if dense:
            loss = -torch.mean((torch.nn.functional.log_softmax(logits, dim=1) * label).sum(1))
        else:
            # perform gradient update, using only ACTIVE neurons as part of sum
            # stop gradient backprop to non-active neurons
            logits = torch.add(logits, softmax_mask)
            log_sm = torch.nn.functional.log_softmax(logits, dim=1)
            # zero out non-active neurons for each sample
            log_sm = torch.multiply(log_sm, active_mask)
            smce = torch.multiply(log_sm, label)

            # stop gradient
            # smce.register_hook(lambda grad: grad * active_mask)

            loss = -torch.mean(torch.sum(smce, 1, keepdim=True))

        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        comp_time = time.time() - init_time

        # compute accuracy
        batch_acc = top1acc(logits, y)
        losses.update(loss_val, train_bs)
        train_acc.update(batch_acc, train_bs)

        # store and save accuracy and loss values
        recorder.add_new(comp_time + lsh_time, comp_time, 0, lsh_time, batch_acc, loss_val)
        recorder.save_to_file()

        # print stats occasionally
        if i % 10 == 0:
            print(
                "Step %d: Epoch Time %f, LSH Time %f, Running Loss %.6f, Running Accuracy %.4f, [%d Total Samples]"
                % (i, (comp_time + lsh_time), lsh_time, losses.avg, train_acc.avg, i*train_bs)
            )
            train_acc.reset()
            losses.reset()

        # validate
        if i % args.steps_per_test == 0 or i % steps_per_epoch == 0:
            p_at_k = 0
            if i % steps_per_epoch == 0:
                test_data_generator = data_generator_test(test_files, test_bs)
                epoch_flag = True
                num_batches = n_test // test_bs
            else:
                epoch_flag = False
                num_batches = args.num_val_batches
            with torch.no_grad():
                for _ in range(num_batches):
                    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
                    print(np.mean(labels_batch[0]))
                    x = torch.sparse_coo_tensor(idxs_batch, vals_batch,
                                                size=(test_bs, nf),
                                                device=device,
                                                requires_grad=False)
                    optimizer.zero_grad()
                    logits = model(x)
                    p_at_k += top1acc_test(logits, labels_batch)

                test_acc = p_at_k / num_batches
                if epoch_flag:
                    epoch = i / steps_per_epoch
                    print("Epoch %d: Top 1 Test Accuracy %.4f\n" % (epoch, test_acc))
                    recorder.add_test_accuracy(test_acc, epoch=epoch_flag)
                else:
                    print("Step %d: Top 1 Test Accuracy %.4f" % (i, test_acc))
                    recorder.add_test_accuracy(test_acc, epoch=epoch_flag)
