import numpy as np
import torch
import time
from misc import top1acc


def pg_train(rank, model, Method, device, optimizer, train_dl, test_dl, losses, train_acc_metric, test_acc_metric,
             recorder, args):
    """
    pg_train: This function orchestrates distributed training of large-scale recommender system training via our
             PGHash algorithm. Everything is written in Python, with no C/C++ code necessary. Some further
             optimizations are required (e.g. reuse partial forward pass during LSH process instead of re-running)

    :param rank: Rank of process
    :param Method: Which algorithm is used to train recommender system (e.g. PGHash or SLIDE)
    :param optimizer: The optimizer performing the gradient updates (e.g. Adam)
    :param train_data: Training data from recommender system dataset
    :param test_data: Test data from recommender system dataset
    :param losses: Function used to compute training loss
    :param train_acc_metric: Metric to store training accuracy values (usually top1)
    :param test_acc_metric: Metric to store test accuracy values (usually top1)
    :param recorder: Recorder to store and write various metrics to file
    :param args: Remaining predefined hyperparameters
    :return: Trained recommender system
    """

    # initialize parameters
    total_batches = 0
    iterations = 1
    test_acc = np.NaN
    comm_time = 0
    comm_time1 = 0
    num_labels = Method.nl
    steps_per_rehash = args.steps_per_lsh
    idx = np.arange(num_labels)

    # begin training below
    for epoch in range(1, args.epochs+1):
        print("\nStart of epoch %d" % (epoch,))

        for data, labels in train_dl:

            '''
            # compute LSH
            lsh_init = time.time()

            # rehashing step: weights of final layer are hashed and placed into buckets depending upon their hash code
            if (iterations-1) % steps_per_rehash == 0:
                Method.rehash()

            # active neuron selection step: each sample in batch is hashed and the resulting hash code is used
            # to select which neurons will be activated (exact matches -- vanilla style)
            active_idx, sample_active_idx, true_neurons_bool = Method.lsh_vanilla(model, data)

            lsh_time = time.time() - lsh_init

            # document the total number of active neurons across the batch
            num_active_neurons = np.count_nonzero(true_neurons_bool)
            total_neruons = 0
            batch_size = len(sample_active_idx)
            for i in range(batch_size):
                total_neruons += len(sample_active_idx[i])
            average_active_per_sample = total_neruons/batch_size
            '''

            data = data.to(device)
            labels = labels.to_dense().to(device)

            # divide by num labels
            labels = labels / torch.count_nonzero(labels, dim=1).unsqueeze(1)

            batch_size, _ = labels.shape
            lsh_time = 0
            average_active_per_sample = 0
            num_active_neurons = 0

            # compute test accuracy every X steps
            if iterations % args.steps_per_test == 0:
                Method.test_accuracy(model, device, test_dl, test_acc_metric, epoch=False)
                test_acc = test_acc_metric.avg
                print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                recorder.add_testacc(test_acc)
                test_acc_metric.reset()

            init_time = time.time()

            '''
            # create a mask to ensure only the active neurons for each sample are used & updated during training
            # note: try to save computational time below by creating mask from indices sans for loop
            mask = np.zeros((batch_size, num_labels))
            for j in range(batch_size):
                mask[j, sample_active_idx[j]] = 1

            active_mask = torch.from_numpy(mask).to(device)
            softmax_mask = torch.where(active_mask == 1, 0., torch.tensor(-1e20)).to(device)

            # perform gradient update, using only ACTIVE neurons as part of sum
            y_pred = model(data)
            # stop gradient backprop to non-active neurons
            # y_pred[:, non_active_idx] = y_pred[:, non_active_idx].detach()
            # y_pred = y_pred[:, active_idx]

            y_pred = torch.add(y_pred, softmax_mask)
            log_sm = torch.nn.functional.log_softmax(y_pred, dim=1)
            # zero out non-active neurons for each sample
            log_sm = torch.multiply(log_sm, active_mask)
            smce = torch.multiply(log_sm, labels)

            # stop gradient
            smce.register_hook(lambda grad: grad * active_mask)

            loss = -torch.mean(torch.sum(smce, 1, keepdim=True))
            '''

            y_pred = model(data)
            loss = -torch.mean(torch.sum(torch.nn.functional.log_softmax(y_pred, dim=1) * labels, dim=1))

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()

            # compute accuracy (top 1) and loss for the minibatch
            rec_init = time.time()
            losses.update(np.array(loss_val), batch_size)
            acc1 = top1acc(y_pred, labels)

            train_acc_metric.update(acc1, batch_size)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - (record_time + comm_time)

            # store and save accuracy and loss values
            recorder.add_new(comp_time + comm_time + lsh_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                             loss_val, train_acc_metric.avg, losses.avg, num_active_neurons, average_active_per_sample)
            recorder.save_to_file()

            # log every X batches
            total_batches += batch_size
            if iterations % 50 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Comm Time %f, LSH Time %f, Loss %.6f, Top 1 Train Accuracy %.4f,"
                    " Average Active Neurons %d, [%d Total Samples]" % (rank, iterations,
                                                                        (comp_time + comm_time + lsh_time),
                                                                        comm_time, lsh_time, loss_val, acc1,
                                                                        average_active_per_sample, total_batches)
                )
            iterations += 1

        Method.test_accuracy(model, device, test_dl, test_acc_metric, epoch=True)
        test_acc = test_acc_metric.avg
        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations - 1, test_acc))
        # recorder.add_testacc(test_acc)
        test_acc_metric.reset()

        # reset accuracy statistics for next epoch
        train_acc_metric.reset()
        losses.reset()
