import tensorflow as tf
import numpy as np
from sparse_bce import sparse_bce, sparse_bce_lsh
from misc import compute_accuracy, compute_accuracy_lsh, AverageMeter, Recorder
from unpack import get_sub_model, get_full_dense, unflatten_weights, flatten_weights, get_model_architecture
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
from mlp import SparseNeuralNetwork
from mpi4py import MPI
import time
import gc


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


def train(rank, model, optimizer, communicator, train_data, test_data, full_model, epochs, gpu, cpu, sdim, num_tables,
          num_f, num_l, hls, cr, lsh, hash_type, steps_per_lsh,
          acc_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=1)):

    # @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            y_true = tf.gather(tf.sparse.to_dense(y), cur_idx, axis=1)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_value, y_pred

    # @tf.function
    def test_step(x, y, cur_idx):
        y_pred = model(x, training=False)
        acc1 = compute_accuracy_lsh(y, y_pred, cur_idx, topk=1)
        return acc1

    top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', rank, hash_type)
    total_batches = 0
    start_idx_w = ((num_f * hls) + hls + (4 * hls))
    start_idx_b = full_model.size - num_l
    used_idx = np.zeros(num_l)
    cur_idx = np.arange(num_l)
    test_acc = np.NaN

    for epoch in range(epochs):

        print("\nStart of epoch %d" % (epoch,))
        with tf.device(cpu):
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_data):

                init_time = time.time()
                batch, s = x_batch_train.get_shape()

                # run lsh here before training if being used
                if lsh:
                    lsh_init_time = time.time()
                    # periodic lsh
                    if step % steps_per_lsh == 0:

                        # compute LSH
                        final_dense = get_full_dense(full_model, num_f, num_l, hls)
                        cur_idx = run_lsh(model, x_batch_train, final_dense, sdim, int(num_tables), cr,
                                          hash_type)
                        used_idx[cur_idx] += 1

                        worker_layer_dims = [num_f, hls, len(cur_idx)]
                        with tf.device(gpu):
                            tf.keras.backend.clear_session()
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

                # compute training step
                loss_value, y_pred = train_step(x_batch_train, y_batch_train)

                # compute accuracy for the minibatch (top 1 and 5) & store accuracy and loss values
                rec_init = time.time()
                acc1 = compute_accuracy_lsh(y_batch_train, y_pred, cur_idx, topk=1)
                losses.update(loss_value.numpy(), batch)
                top1.update(acc1, batch)
                record_time = time.time() - rec_init

                comp_time = (time.time() - init_time) - (lsh_time + record_time)

                # communication happens here
                if lsh:
                    full_model, comm_time = communicator.communicate(model, full_model, cur_idx, start_idx_b,
                                                                     layer_shapes, layer_sizes)
                else:
                    comm_time = communicator.communicate(model)
                    full_model = flatten_weights(model.get_weights())

                total_batches += batch
                # Log every 10 batches.
                if step % 10 == 0:
                    print(
                        "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Accuracy %.4f, [%d Total Samples]" % (
                        rank, step, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                    )

                if step % 5 == 0:
                    if rank == 0:
                        top1_test = AverageMeter()
                        with tf.device(cpu):
                            t = time.time()
                            for step, (x_batch_test, y_batch_test) in enumerate(test_data):
                                #test_step(x_batch_test, tf.sparse.to_dense(y_batch_test), None)
                                acc = test_step(x_batch_test, y_batch_test, cur_idx)
                                top1_test.update(acc, x_batch_test.get_shape()[0])
                            test_acc = top1_test.avg
                            print("Test Accuracy Top 1: %.4f In %f seconds" % (test_acc, time.time()-t))
                            # print("Test Accuracy Top 1: %.4f" % (float(acc_metric.result().numpy()),))
                            # acc_metric.reset_state()

                MPI.COMM_WORLD.Barrier()
                recorder.add_new(comp_time + comm_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                                 loss_value.numpy(), top1.avg, losses.avg)
                test_acc = np.NaN


        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
        # Save data to output folder
        recorder.save_to_file()

        '''
        if rank == 0:
            with tf.device(cpu):
                worker_layer_dims = [num_f, hls, num_l]
                model = SparseNeuralNetwork(worker_layer_dims)
                layer_shapes, layer_sizes = get_model_architecture(model)
                # set new sub-model
                w, b = get_sub_model(full_model, np.arange(num_l), start_idx_b, num_f, hls)
                sub_model = np.concatenate((full_model[:start_idx_w], w.flatten(), b.flatten()))
                new_weights = unflatten_weights(sub_model, layer_shapes, layer_sizes)
                model.set_weights(new_weights)
                for step, (x_batch_test, y_batch_test) in enumerate(test_data):
                    test_step(x_batch_test, tf.sparse.to_dense(y_batch_test), None)
                print("Test Accuracy Top 1: %.4f" % (float(acc_metric.result().numpy()),))
        '''

    # Run a test loop at the end of training
    print('Testing Model...')
    with tf.device(cpu):
        for step, (x_batch_test, y_batch_test) in enumerate(test_data):
            test_step(x_batch_test, tf.sparse.to_dense(y_batch_test), None)
        print("Test Accuracy Top 1: %.4f" % (float(acc_metric.result().numpy()),))

    return full_model, used_idx, recorder.get_saveFolder()
