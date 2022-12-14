import tensorflow as tf
import numpy as np
from misc import compute_accuracy, AverageMeter, Recorder
from unpack import get_sub_model, get_full_dense, get_model_architecture, unflatten_weights
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
from mlp import SparseNeuralNetwork
from mpi4py import MPI
import time


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


def train(rank, model, optimizer, communicator, train_data, test_data, full_model, epochs, sdim, num_tables,
          num_f, num_l, hls, cr, lsh, hash_type, steps_per_lsh,
          acc_metric=tf.keras.metrics.TopKCategoricalAccuracy(k=1)):

    # @tf.function
    def train_step(x, y, idx):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            y_true = tf.gather(tf.sparse.to_dense(y), idx, axis=1)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # del (y_true)
        # del (grads)
        # gc.collect()
        return loss_value, y_pred

    # @tf.function
    def test_step(x, y, cur_idx):
        y_pred = model(x, training=False)
        acc_metric.update_state(y_pred, y)
        # del (y_pred)
        # gc.collect()
        # acc1 = compute_accuracy(y, y_pred, cur_idx, topk=1)
        # bs = x.get_shape()[0]
        # return acc1, bs

    top1 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', MPI.COMM_WORLD.Get_size(), rank, hash_type)
    total_batches = 0
    start_idx_b = full_model.size - num_l
    start_idx_w = ((num_f * hls) + hls + (4 * hls))
    used_idx = np.zeros(num_l)
    cur_idx = np.arange(num_l)
    test_acc = np.NaN

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # if epoch == 1:
        #     steps_per_lsh = 50
        # elif epoch == 2:
        #    steps_per_lsh = 100

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

            loss_value, y_pred = train_step(x_batch_train, y_batch_train, cur_idx)

            # compute accuracy for the minibatch (top 1) & store accuracy and loss values
            rec_init = time.time()
            losses.update(np.array(loss_value), batch)
            acc1 = compute_accuracy(y_batch_train, y_pred, cur_idx, topk=1)
            top1.update(acc1, batch)
            record_time = time.time() - rec_init
            comp_time = (time.time() - init_time) - (lsh_time + record_time)

            # communication happens here
            if lsh:
                full_model, comm_time = communicator.communicate(model, full_model, cur_idx, start_idx_b,
                                                                 layer_shapes, layer_sizes)
            else:
                comm_time = communicator.communicate(model)

            recorder.add_new(comp_time+comm_time, comp_time, comm_time, lsh_time, acc1, test_acc, loss_value.numpy(),
                             top1.avg, losses.avg)

            # Save data to output folder
            recorder.save_to_file()

            total_batches += batch
            # Log every 200 batches.
            if step % 10 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Accuracy %.4f, [%d Total Samples]"
                    % (rank, step, (comp_time + comm_time), loss_value.numpy(), acc1, total_batches)
                )

            '''
            if step % 100 == 0:
                if rank == 0:
                    #model.save_weights('./Output/my_checkpoint-step'+str(step))
                    # test_top1 = AverageMeter()
                    #for step, (x_batch_test, y_batch_test) in enumerate(test_data):
                    #    test_step(x_batch_test, tf.sparse.to_dense(y_batch_test), None)
                        # y_pred = model.predict_on_batch(x_batch_test)
                        # y_pred = tf.convert_to_tensor(y_pred)
                        # acc1 = compute_accuracy(y_batch_train, y_pred, cur_idx, topk=1, numpy=True)
                        # test_top1.update(acc1, batch)
                        # del (y_pred)
                        # gc.collect()
                        # acc_metric.update_state(y_pred, tf.sparse.to_dense(y_batch_test))
                    #print("Test Accuracy Top 1: %.4f" % (float(acc_metric.result().numpy()),))
                    # print("Test Accuracy Top 1: %.4f" % (float(test_top1.avg),))
                    #acc_metric.reset_state()

            '''

        # reset accuracy statistics for next epoch
        top1.reset()
        # top5.reset()
        losses.reset()


    # Run a test loop at the end of training
    print('Testing Model...')
    for step, (x_batch_test, y_batch_test) in enumerate(test_data):
        test_step(x_batch_test, y_batch_test, cur_idx)
    print("Test Accuracy Top 1: %.4f" % (float(acc_metric.result().numpy()),))

    return full_model, used_idx, recorder.get_saveFolder()
