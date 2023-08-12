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
    parser.add_argument('--num_val_batches', type=int, default=50)
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
    if args.dataset == 'Amazon670K':
        nc = 670091
        nf = 135909
        n_train = 490449
        n_test = 153025

    elif args.dataset == 'Delicious200K':
        nc = 205443
        nf = 782585
        n_train = 196606
        n_test = 100095

    steps_per_epoch = 196606 // args.train_bs
    n_steps = args.epochs * steps_per_epoch

    if k > c and args.dwta == 1:
        print('Error: Compression Size Smaller than Hash Length for PGHash-D')
        exit()

    if args.hash_type[:2] == 'pg':
        method = 'PGHash'
        cr = args.cr
        slide = False
    else:
        method = 'SLIDE'
        cr = 1
        slide = True

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

    lsh_time = 0
    for i in range(1, n_steps+1):
        # train
        idxs_batch, vals_batch, y = next(training_data_generator)
        x = torch.sparse_coo_tensor(idxs_batch, vals_batch,
                                    size=(train_bs, nf),
                                    device=device,
                                    requires_grad=False)
        label = torch.from_numpy(y).to(device)
        optimizer.zero_grad()

        init_time = time.time()
        logits = model(x)
        loss = -torch.mean((torch.nn.functional.log_softmax(logits, dim=1) * label).sum(1))
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
            test_data_generator = data_generator_test(test_files, test_bs)
            p_at_k = 0
            if i % steps_per_epoch == 0:
                epoch_flag = True
                num_batches = n_test // test_bs
            else:
                epoch_flag = False
                num_batches = args.num_val_batches
            with torch.no_grad():
                for _ in range(num_batches):
                    idxs_batch, vals_batch, labels_batch = next(test_data_generator)
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
