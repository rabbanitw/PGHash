import numpy as np
import tensorflow as tf
import time
from util.misc import compute_accuracy_lsh


def get_partial_label_mask(sparse_y, sub_idx, sample_idx, batch_size, args, idx):

    y_true = tf.sparse.to_dense(sparse_y)

    # make sure all samples are divided by number of labels
    nz = tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
    y_true = y_true / nz

    # TRY TO CREATE MASK FROM INDICES WITHOUT HAVING TO USE FORLOOP (SAVE 0.025 seconds)
    num_labels = y_true.shape[1]
    mask = np.zeros((batch_size, num_labels))
    for j in range(batch_size):
        mask[j, sample_idx[j + idx*args.train_bs]] = 1
    mask = mask[:, sub_idx]

    # shorten the true label
    y_true = tf.gather(y_true, indices=sub_idx, axis=1)

    leftout_labels = nz - tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)

    return y_true, leftout_labels, 1/nz, tf.convert_to_tensor(mask, dtype=tf.dtypes.float32)


def pg_train(rank, size, Method, optimizer, train_data, test_data, losses, top1, test_top1, recorder, args):

    # parameters
    total_batches = 0
    iterations = 1
    test_acc = np.NaN
    comm_time = 0
    if args.cr == 1:
        smartavg = False
    else:
        smartavg = True

    num_labels = Method.nl
    num_features = Method.nf

    # nl_tf = tf.constant(num_labels, dtype=tf.int32)
    num_diff = tf.constant(num_labels - Method.num_c_layers, dtype=tf.float32)
    for epoch in range(1, args.epochs+1):
        print("\nStart of epoch %d" % (epoch,))

        # shuffle training data each epoch
        train_data.shuffle(len(train_data))

        # iterate over the batches of the dataset.
        for (x_batch_train, y_batch_train) in train_data:

            batches_per_q = np.ceil(x_batch_train.shape[0] / args.train_bs).astype(np.int32)

            # update full model
            Method.update_full_model(Method.model)

            # compute LSH
            lsh_init = time.time()
            active_idx, sample_active_idx = Method.lsh_hamming(Method.model, x_batch_train)
            # active_idx, sample_active_idx = Method.lsh_avg_hamming(Method.model, x_batch_train)
            lsh_time = time.time() - lsh_init

            # testing random training
            # active_idx = np.sort(np.random.choice(np.arange(num_labels), size=Method.num_c_layers, replace=False))
            # active_idx = np.arange(Method.num_c_layers)
            # sample_active_idx = [active_idx for _ in range(args.train_bs)]
            # Method.ci = active_idx
            # update indices with new current index
            # Method.bias_idx = Method.ci + Method.bias_start

            if size > 1:
                # send indices to root (server)
                Method.exchange_idx()

            # update model
            Method.update_model()

            for sub_batch_idx in range(batches_per_q):

                # compute test accuracy every X steps
                if iterations % args.steps_per_test == 0:
                    if rank == 0:
                        Method.update_full_model(Method.model)
                        test_acc = Method.test_full_model(test_data, test_top1)
                        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                        recorder.add_testacc(test_acc)
                        test_top1.reset()

                x = tf.sparse.slice(x_batch_train, start=[sub_batch_idx * args.train_bs, 0],
                                    size=[args.train_bs, num_features])
                y = tf.sparse.slice(y_batch_train, start=[sub_batch_idx * args.train_bs, 0],
                                    size=[args.train_bs, num_labels])

                # don't count LSH time towards subsequent batches of the mega batch_size
                if sub_batch_idx == 1:
                    lsh_time = 0

                init_time = time.time()

                # communicate models amongst devices (if multiple devices are present)
                if size > 1:
                    model, comm_time = Method.communicate(Method.model, smart=smartavg)

                batch_size = x.get_shape()[0]

                '''
                full_mask = np.zeros((batch_size, num_labels))
                inverse_full_mask = np.ones((batch_size, num_labels))
                for j in range(batch_size):
                    full_mask[j, sample_active_idx[j + sub_batch_idx * args.train_bs]] = 1
                    # inverse_full_mask[j, sample_active_idx[j + sub_batch_idx * args.train_bs]] = 0
                sub_mask = full_mask[:, active_idx]
                sub_mask = tf.convert_to_tensor(sub_mask, dtype=tf.dtypes.float32)
                inverse_full_mask = tf.convert_to_tensor(inverse_full_mask, dtype=tf.dtypes.float32)
                # full_mask = tf.convert_to_tensor(full_mask, dtype=tf.dtypes.float32)
                active_mask2 = tf.where(sub_mask == 1, 0., tf.float32.min / 100)

                active_indices = [[x] for x in active_idx]
                shape = tf.constant([num_labels, batch_size])
                y_true = tf.sparse.to_dense(y)
                # make each row a valid probability distribution
                nz = tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
                y_true = y_true / nz

                zero = tf.constant(0, dtype=tf.float32)
                mask = tf.not_equal(y_true, zero)
                full_mask = tf.where(mask, 0., tf.float32.min)
                mask = tf.where(mask, 1., 0.)

                # get biases for loss (since they shouldn't be fully zero'd out)
                full_biases = Method.full_model[Method.bias_start:]
                bias = tf.math.multiply(mask, inverse_full_mask) * full_biases
                '''

                '''
                full_mask = np.ones((batch_size, num_labels)) * (np.finfo(np.float32).min/100)
                full_mask2 = np.zeros((batch_size, num_labels))
                for j in range(batch_size):
                    full_mask[j, sample_active_idx[j + sub_batch_idx * args.train_bs]] = 0
                    full_mask2[j, sample_active_idx[j + sub_batch_idx * args.train_bs]] = 1
                full_mask = tf.convert_to_tensor(full_mask, dtype=tf.dtypes.float32)
                full_mask2 = tf.convert_to_tensor(full_mask2, dtype=tf.dtypes.float32)
                '''

                '''
                active_indices = [[x] for x in active_idx]
                shape = tf.constant([num_labels, batch_size])
                y_true = tf.sparse.to_dense(y)
                # make each row a valid probability distribution
                nz = tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
                y_true = y_true / nz
                '''

                #full_mask = np.zeros((batch_size, num_labels))
                #for j in range(batch_size):
                #    full_mask[j, sample_active_idx[j + sub_batch_idx * args.train_bs]] = 1
                #mask = tf.convert_to_tensor(full_mask[:, active_idx], dtype=tf.dtypes.float32)

                y_true, leftover, label_frac, active_mask = get_partial_label_mask(y, active_idx, sample_active_idx, batch_size, args, sub_batch_idx)
                active_mask2 = tf.where(active_mask == 1, 0., tf.float32.min / 10)

                # perform gradient update
                with tf.GradientTape() as tape:

                    '''
                    # This computes loss over only TRUE label indices
                    y_pred = Method.model(x)
                    y_pred = tf.math.multiply(y_pred, sub_mask)
                    y_pred_full = tf.transpose(tf.scatter_nd(tf.constant(active_indices), tf.transpose(y_pred), shape))
                    y_pred_full = y_pred_full + full_mask + bias
                    loss_value = -tf.reduce_mean(tf.reduce_sum(tf.math.multiply(tf.nn.log_softmax(y_pred_full, axis=1), mask) * y_true, axis=1))
                    '''

                    '''
                    y_pred = Method.model(x)
                    y_pred2 = tf.math.add(y_pred, active_mask2)
                    max_logit = tf.math.reduce_max(y_pred2, axis=1, keepdims=True)
                    # inner exponential sum
                    pred_exp = tf.math.exp(y_pred2 - max_logit)
                    log_exp_sum = tf.math.log(tf.reduce_sum(pred_exp, axis=1, keepdims=True))

                    y_pred_full = tf.transpose(tf.scatter_nd(tf.constant(active_indices), tf.transpose(y_pred), shape))
                    y_pred_full2 = y_pred_full + full_mask + bias
                    y_pred_full2 = y_pred_full2 - max_logit - log_exp_sum
                    loss_value = -tf.reduce_mean(tf.reduce_sum(tf.math.multiply(y_pred_full2, mask) * y_true, axis=1))
                    '''

                    #'''
                    # This is custom using only ACTIVE neurons as part of sum. Good results so far
                    y_pred = Method.model(x)
                    # y_pred = tf.math.multiply(y_pred, active_mask) # need to come up with better here
                    y_pred = tf.math.add(y_pred, active_mask2)
                    # ==== custom loss ====
                    max_logit = tf.math.reduce_max(y_pred, axis=1, keepdims=True)
                    # inner exponential sum
                    pred_exp = tf.math.exp(y_pred - max_logit)
                    # pred_exp = tf.math.multiply(pred_exp, active_mask)
                    exp_sum = tf.reduce_sum(pred_exp, axis=1, keepdims=True)
                    log_sm = y_pred - max_logit - tf.math.log(exp_sum)
                    # inner loss (using mask)
                    log_sm = tf.math.multiply(log_sm, active_mask)
                    inner = tf.reduce_sum(tf.math.multiply(log_sm, y_true), axis=1, keepdims=True)
                    # outer loss
                    outer = leftover * label_frac * tf.math.log(1e-12)
                    loss_value = -tf.reduce_mean(inner + outer)
                    #'''

                    '''
                    # THIS IS CUSTOM USING OUTSIDE AS PART OF SUM
                    y_pred = Method.model(x)
                    y_pred = tf.math.multiply(y_pred, mask) # + active_bias
                    # ==== custom loss ====
                    max_logit = tf.math.maximum(tf.math.reduce_max(y_pred, axis=1, keepdims=True), 0)
                    # inner exponential sum
                    inner_exp_sum = tf.reduce_sum(tf.math.exp(y_pred - max_logit), axis=1, keepdims=True)
                    # outer exponential sum
                    outside_e_logit = tf.math.exp(-max_logit)
                    outer_exp_sum = num_diff * outside_e_logit
                    # sum of inner and outer
                    e_sum = inner_exp_sum + outer_exp_sum
                    # log of inner and outer sum
                    log_sum_exp = tf.math.log(e_sum)
                    log_sm = y_pred - max_logit - log_sum_exp
                    # inner loss (using mask)
                    inner = tf.reduce_sum(tf.math.multiply(log_sm, y_true), axis=1, keepdims=True)
                    # outer loss
                    outer = -leftover * label_frac * (max_logit + log_sum_exp)
                    loss_value = -tf.reduce_mean(inner + outer)
                    '''

                    # This uses outside as sum, works when we use fixed neurons for cr=0.99
                    # y_pred = Method.model(x)
                    # y_pred_full = tf.transpose(tf.scatter_nd(tf.constant(active_indices), tf.transpose(y_pred), shape))
                    # y_pred_full = tf.math.multiply(y_pred_full, full_mask)
                    # loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred_full))

                grads = tape.gradient(loss_value, Method.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, Method.model.trainable_weights))

                # compute accuracy (top 1) and loss for the minibatch
                rec_init = time.time()
                losses.update(np.array(loss_value), batch_size)
                acc1 = compute_accuracy_lsh(y_pred, y, active_idx, num_labels)
                top1.update(acc1, batch_size)
                record_time = time.time() - rec_init
                comp_time = (time.time() - init_time) - (record_time + comm_time)

                if sub_batch_idx != 0:
                    lsh_time = 0

                # store and save accuracy and loss values
                recorder.add_new(comp_time + comm_time + lsh_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                                 loss_value.numpy(), top1.avg, losses.avg)
                recorder.save_to_file()

                # log every X batches
                total_batches += batch_size
                if iterations % 5 == 0:
                    print(
                        "(Rank %d) Step %d: Epoch Time %f, Comm Time %f, LSH Time %f, Loss %.6f, Top 1 Train Accuracy %.4f, "
                        "[%d Total Samples]" % (rank, iterations, (comp_time + comm_time), comm_time, lsh_time,
                                                loss_value.numpy(), acc1, total_batches)
                    )
                iterations += 1

        # reset accuracy statistics for next epoch
        top1.reset()
        losses.reset()
