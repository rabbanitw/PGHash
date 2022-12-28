import tensorflow as tf
import numpy as np
from sparse_bce import sparse_bce, sparse_bce_lsh
from misc import compute_accuracy, compute_accuracy_lsh, AverageMeter, Recorder
from unpack import get_sub_model, get_full_dense, get_model_architecture, unflatten_weights
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
from mlp import SparseNeuralNetwork
from mpi4py import MPI
import time

import resource
import os
import datetime


class SparseCE_Loss(tf.keras.losses.Loss):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
    def call(self, y_true, y_pred):
        y_true = tf.gather(tf.sparse.to_dense(y_true), self.idx, axis=1)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def run_lsh(model, data, final_dense_w, sdim, num_tables, cr, hash_type):

    # get input layer for LSH
    feature_extractor = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.layers[-3].output,
    )
    in_layer = feature_extractor(data).numpy()

    # run LSH to find the most important weights
    if hash_type == "pg_vanilla":
        return pg_vanilla(in_layer, final_dense_w, sdim, num_tables, cr)
    elif hash_type == "pg_avg":
        return pg_avg(in_layer, final_dense_w, sdim, num_tables, cr)
    elif hash_type == "slide_vanilla":
        return slide_vanilla(in_layer, final_dense_w, sdim, num_tables, cr)
    elif hash_type == "slide_avg":
        return slide_avg(in_layer, final_dense_w, sdim, num_tables, cr)


def train(rank, model, optimizer, communicator, train_data, test_data, full_model,
          num_f, num_l, args, acc_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=1)):

    # @tf.function
    def train_step(x, y):
        # y = tf.gather(tf.sparse.to_dense(y), idx, axis=1)
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, y_pred

    # @tf.function
    def test_step(x, y, cur_idx):
        y_pred = model(x, training=False)
        acc_metric.update_state(y_pred, y)

    def get_memory(filename):
        mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        with open(filename, 'a') as f:
            # Dump timestamp, PID and amount of RAM.
            f.write('{} {} {}\n'.format(datetime.datetime.now(), os.getpid(), mem))

    def get_partial_label(y_data, used_idx, batch_size, full_labels):
        true_idx = y_data.indices.numpy()
        y_true = np.zeros((batch_size, full_labels))
        for i in true_idx:
            y_true[i[0], i[1]] = 1
        get_memory(fname)
        # return y_true[:, used_idx]
        return tf.convert_to_tensor(y_true[:, used_idx])

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
    start_idx_b = full_model.size - num_l
    start_idx_w = ((num_f * hls) + hls + (4 * hls))
    used_idx = np.zeros(num_l)
    cur_idx = np.arange(num_l)
    test_acc = np.NaN

    cur_idx = tf.convert_to_tensor(cur_idx)

    print(model.summary())

    fname = 'r{}.log'.format(rank)
    if os.path.exists(fname):
        os.remove(fname)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_data):
            lsh_time = 0
            init_time = time.time()

            '''
            # run lsh here before training if being used
            if lsh:
                lsh_init_time = time.time()
                # periodic lsh
                if step % steps_per_lsh == 0:

                    # compute LSH
                    final_dense = get_full_dense(full_model, num_f, num_l, hls)
                    cur_idx = run_lsh(model, x_batch_train, final_dense, sdim, int(num_tables), cr, hash_type)
                    used_idx[cur_idx] += 1
                    # print(num_l)

                    worker_layer_dims = [num_f, hls, len(cur_idx)]
                    model = SparseNeuralNetwork(worker_layer_dims)
                    layer_shapes, layer_sizes = get_model_architecture(model)

                    # set new sub-model
                    w, b = get_sub_model(full_model, cur_idx, start_idx_b, num_f, hls)
                    sub_model = np.concatenate((full_model[:start_idx_w], w.flatten(), b.flatten()))
                    new_weights = unflatten_weights(sub_model, layer_shapes, layer_sizes)
                    model.set_weights(new_weights)

                lsh_time = time.time()-lsh_init_time
            else:
                lsh_time = 0
            '''
            '''
            if lsh:
                full_model, comm_time = communicator.communicate(model, full_model, cur_idx, start_idx_b,
                                                                 layer_shapes, layer_sizes)
            else:
                comm_time = communicator.communicate(model)
            '''

            batch, s = x_batch_train.get_shape()
            get_memory(fname)
            y_true = get_partial_label(y_batch_train, cur_idx, batch, num_l)

            '''
            get_memory(fname)
            with tf.GradientTape() as tape:
                y_pred = model(x_batch_train, training=True)
                get_memory(fname)
                loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
                get_memory(fname)
            grads = tape.gradient(loss_value, model.trainable_weights)
            get_memory(fname)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            get_memory(fname)

            with open(fname, 'a') as f:
                # Dump timestamp, PID and amount of RAM.
                f.write('==========\n')

            '''
            # '''
            # batch, s = x_batch_train.get_shape()
            lsh_time = 0
            loss_value, y_pred = train_step(x_batch_train, y_true)
            # '''

            
            # compute accuracy for the minibatch (top 1) & store accuracy and loss values
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)

            # NEED TO FIX THIS ACCURACY METRIC
            # acc1 = compute_accuracy(y_batch_train, y_pred, cur_idx, topk=1)

            acc1 = acc_metric(y_pred, tf.sparse.to_dense(y_batch_train))
            top1.update(acc1, batch)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - (lsh_time + record_time)
            

            # communication happens here
            # comm_time = communicator.communicate(model)
            comm_time = 0

            recorder.add_new(comp_time+comm_time, comp_time, comm_time, lsh_time, acc1, test_acc, loss_value.numpy(),
                             top1.avg, losses.avg)

            # Save data to output folder
            recorder.save_to_file()
            #'''

            total_batches += batch
            # Log every 200 batches.
            if step % 10 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Accuracy %.4f, [%d Total Samples]"
                    % (rank, step, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                )

        # reset accuracy statistics for next epoch
        top1.reset()
        # top5.reset()
        losses.reset()

    return full_model, used_idx, recorder.get_saveFolder()
