import numpy as np
from misc import compute_accuracy, compute_accuracy_lsh, AverageMeter, Recorder
# from unpack import get_sub_model, get_full_dense, get_model_architecture, unflatten_weights
# from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
# from mlp import SparseNeuralNetwork
from mpi4py import MPI
import torch
import time
from mlp import NeuralNetwork
import resource
import os
import datetime
from typing import List


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        _, target = target.topk(k=maxk, dim=1)
        target = target.t()
        #print(target.shape)
        #target = target.flatten()


        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        # target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]

        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


def train(rank, model, optimizer, train_data, test_data, num_f, num_l, args):

    def get_memory(filename):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(filename, 'a') as f:
            # Dump timestamp, PID and amount of RAM.
            f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

    def softmax_cross_entropy_with_logits(logits, labels, dim=-1):
        return (-labels * torch.nn.functional.log_softmax(logits, dim=dim)).sum(dim=dim)

    def BCEwithLogits():
        loss = 0
        return loss

    # hashing parameters
    sdim = args.sdim
    num_tables = args.num_tables
    cr = args.cr
    lsh = args.lsh
    hash_type = args.hash_type
    steps_per_lsh = args.steps_per_lsh

    # training parameters
    epochs = args.epochs
    hls = args.hidden_layer_size

    top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', args.name, MPI.COMM_WORLD.Get_size(), rank, hash_type)
    total_batches = 0
    used_idx = np.zeros(num_l)
    cur_idx = np.arange(num_l)
    test_acc = np.NaN

    fname = 'r{}.log'.format(rank)
    if os.path.exists(fname):
        os.remove(fname)

    # model = NeuralNetwork(num_f, hls, num_l)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    model.train(True)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):


            lsh_time = 0
            init_time = time.time()
            batch = x_batch_train.size(0)

            # '''
            get_memory(fname)

            # Make predictions for this batch
            outputs = model(x_batch_train.to_dense())
            get_memory(fname)

            # Compute the loss and its gradients
            y_true = y_batch_train.to_dense()
            # loss = torch.mean(softmax_cross_entropy_with_logits(outputs, y_true))
            loss = criterion(outputs, y_true)

            get_memory(fname)
            # Adjust learning weights
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            get_memory(fname)

            rec_init = time.time()
            # Gather data and report
            loss_value = loss.item()
            losses.update(loss_value, batch)
            # Compute accuracy
            acc1 = accuracy(outputs, y_true, topk=(1,))[0].numpy()
            top1.update(acc1, batch)
            record_time = time.time() - rec_init

            comp_time = (time.time() - init_time) - (lsh_time + record_time)

            # '''

            with open(fname, 'a') as f:
                # Dump timestamp, PID and amount of RAM.
                f.write('==========\n')

            '''
            # communication happens here
            # comm_time = communicator.communicate(model)
            '''

            comm_time = 0

            recorder.add_new(comp_time + comm_time, comp_time, comm_time, lsh_time, acc1, test_acc, loss_value,
                             top1.avg, losses.avg)

            # Save data to output folder
            recorder.save_to_file()

            total_batches += batch
            # Log every 200 batches.
            if step % 10 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Accuracy %.4f, [%d Total Samples]"
                    % (rank, step, (comp_time + comm_time), loss_value, acc1, total_batches)
                )


        # reset accuracy statistics for next epoch
        top1.reset()
        # top5.reset()
        losses.reset()

    return None, used_idx, recorder.get_saveFolder()
