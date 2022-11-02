import tensorflow as tf
import numpy as np
from sparse_bce import sparse_bce, sparse_bce_lsh
from misc import compute_accuracy, compute_accuracy_lsh, AverageMeter, Recorder
from unpack import get_sub_model, get_full_dense, get_model_architecture, unflatten_weights
from lsh import pg_avg, pg_vanilla, slide_avg, slide_vanilla
from mlp import SparseNeuralNetwork
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
          num_f, num_l, hls, cr, lsh, hash_type, steps_per_lsh):

    top1 = AverageMeter()
    top5 = AverageMeter()
    losses = AverageMeter()
    recorder = Recorder('Output', rank, hash_type)
    total_batches = 0
    start_idx_b = full_model.size - num_l
    start_idx_w = ((num_f * hls) + hls + (4 * hls))
    used_idx = np.zeros(num_l)
    cur_idx = None

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        if epoch == 1:
            steps_per_lsh = 50
        elif epoch == 2:
            steps_per_lsh = 100

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
                    print(len(cur_idx))
                    print(num_l)

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

            with tf.GradientTape() as tape:
                # Run the forward pass of the layer.
                # The operations that the layer applies
                # to its inputs are going to be recorded
                # on the GradientTape.


                y_pred = model(x_batch_train, training=True)

                # Compute the loss value for this minibatch.
                if lsh:
                    y_true = tf.gather(tf.sparse.to_dense(y_batch_train), cur_idx, axis=1)
                    loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
                    # loss_value = sparse_bce_lsh(y_batch_train, y_pred, cur_idx)
                else:
                    loss_value = sparse_bce(y_batch_train, y_pred)

                # compute accuracy for the minibatch (top 1 and 5) & store accuracy and loss values
                rec_init = time.time()
                acc1 = compute_accuracy_lsh(y_batch_train, y_pred, cur_idx, topk=1)
                acc5 = compute_accuracy_lsh(y_batch_train, y_pred, cur_idx, topk=5)
                losses.update(loss_value.numpy(), batch)
                top1.update(acc1, batch)
                top5.update(acc5, batch)
                record_time = time.time() - rec_init

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            comp_time = (time.time() - init_time) - (lsh_time + record_time)

            # communication happens here
            if lsh:
                full_model, comm_time = communicator.communicate(model, full_model, cur_idx, start_idx_b,
                                                                 layer_shapes, layer_sizes)
            else:
                comm_time = communicator.communicate(model)

            recorder.add_new(comp_time+comm_time, comp_time, comm_time, lsh_time, acc1, acc5, loss_value.numpy(),
                             top1.avg, top5.avg, losses.avg)
            total_batches += batch
            # Log every 200 batches.
            if step % 5 == 0:
                print(
                    "(Rank %d) Step %d: Epoch Time %f, Loss %.6f, Top 1 Accuracy %.4f, "
                    "Top 5 Accuracy %.4f [%d Total Samples]" % (rank, step, (comp_time + comm_time), loss_value.numpy(),
                                                                acc1, acc5, total_batches)
                )

        # reset accuracy statistics for next epoch
        top1.reset()
        top5.reset()
        losses.reset()
        # Save data to output folder
        recorder.save_to_file()

        if rank == 0:
            test_top1 = AverageMeter()
            test_top5 = AverageMeter()
            worker_layer_dims = [num_f, hls, num_l]
            model = SparseNeuralNetwork(worker_layer_dims)
            layer_shapes, layer_sizes = get_model_architecture(model)
            # set new sub-model
            w, b = get_sub_model(full_model, np.arange(num_l), start_idx_b, num_f, hls)
            sub_model = np.concatenate((full_model[:start_idx_w], w.flatten(), b.flatten()))
            new_weights = unflatten_weights(sub_model, layer_shapes, layer_sizes)
            model.set_weights(new_weights)
            for step, (x_batch_test, y_batch_test) in enumerate(test_data):
                y_pred = model(x_batch_test, training=False)
                acc1 = compute_accuracy_lsh(y_batch_test, y_pred, np.arange(num_l), topk=1)
                acc5 = compute_accuracy_lsh(y_batch_test, y_pred, np.arange(num_l), topk=5)
                bs = x_batch_test.get_shape()[0]
                test_top1.update(acc1, bs)
                test_top5.update(acc5, bs)
            print("Test Accuracy Top 1: %.4f" % (float(test_top1.avg),))
            print("Test Accuracy Top 5: %.4f" % (float(test_top5.avg),))

    # Run a test loop at the end of training
    test_top1 = AverageMeter()
    test_top5 = AverageMeter()
    print('Testing Model...')
    for step, (x_batch_test, y_batch_test) in enumerate(test_data):
        y_pred = model(x_batch_test, training=False)
        # Update test metrics
        acc1 = compute_accuracy_lsh(y_batch_test, y_pred, cur_idx, topk=1)
        acc5 = compute_accuracy_lsh(y_batch_test, y_pred, cur_idx, topk=5)
        bs = x_batch_test.get_shape()[0]
        test_top1.update(acc1, bs)
        test_top5.update(acc5, bs)
    print("Test Accuracy Top 1: %.4f" % (float(test_top1.avg),))
    print("Test Accuracy Top 5: %.4f" % (float(test_top5.avg),))

    return full_model, used_idx, recorder.get_saveFolder()
