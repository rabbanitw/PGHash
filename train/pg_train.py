import numpy as np
import tensorflow as tf
import time
from util.misc import compute_accuracy_lsh


class inner_softmax_ce(tf.keras.losses.Loss):
    # initialize instance attributes
    def __init__(self, num_diff):
        super(inner_softmax_ce, self).__init__()
        self.num_diff = num_diff

    # Compute loss
    def call(self, y_true, y_pred):
        max_logit = tf.math.maximum(tf.math.reduce_max(y_pred, axis=1, keepdims=True), 0)
        # inner exponential sum
        inner_exp_sum = tf.reduce_sum(tf.math.exp(y_pred - max_logit), axis=1, keepdims=True)
        # outer exponential sum
        outside_e_logit = tf.math.maximum(tf.math.exp(-max_logit), 1e-12)
        outer_exp_sum = self.num_diff * outside_e_logit
        # sum of inner and outer
        e_sum = inner_exp_sum + outer_exp_sum
        # log of inner and outer sum
        log_sum_exp = tf.math.log(e_sum)
        log_sm = y_pred - max_logit - log_sum_exp
        # inner loss (using mask)
        return tf.reduce_sum(tf.math.multiply(log_sm, y_true), axis=1, keepdims=True)


def get_partial_label_mask(sparse_y, sub_idx, sample_idx, batch_size, args, idx):

    t = time.time()
    y_true = tf.sparse.to_dense(sparse_y)
    #print(time.time()-t)

    # make sure all samples are divided by number of labels
    nz = tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
    y_true = y_true / nz
    #print(time.time() - t)

    # TRY TO CREATE MASK FROM INDICES WITHOUT HAVING TO USE FORLOOP (SAVE 0.025 seconds)
    mask = np.zeros((batch_size, y_true.shape[1]))
    for j in range(batch_size):
        mask[j, sample_idx[j + idx*args.train_bs]] = 1
        # mask[j, sample_idx[j + idx*args.train_bs, :]] = 1
    #print(time.time() - t)

    # mask the true label
    y_true = y_true * mask

    # shorten the true label
    y_true = tf.gather(y_true, indices=sub_idx, axis=1)

    leftout_labels = nz - tf.math.count_nonzero(y_true, axis=1, dtype=tf.dtypes.float32, keepdims=True)
    #print(time.time() - t)

    return y_true, leftout_labels, 1/nz


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
    num_diff = tf.constant(num_labels - Method.num_c_layers, dtype=tf.float32)
    lossF = inner_softmax_ce(num_diff)

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

            # possible LSH methods, Hamming is superior
            # if iterations % args.steps_per_lsh == 0 or iterations == 1:
            #    Method.lsh_tables()
            # cur_idx, per_sample_idx = Method.lsh(Method.model, x_batch_train)
            # cur_idx, per_sample_idx = Method.lsh_initial(Method.model, x_batch_train)
            # cur_idx, per_sample_idx = Method.lsh_vanilla(Method.model, x_batch_train)

            lsh_init = time.time()
            cur_idx, per_sample_idx = Method.lsh_hamming(Method.model, x_batch_train)
            lsh_time = time.time() - lsh_init

            if size > 1:
                # send indices to root (server)
                Method.exchange_idx()
            # update model
            Method.update_model()

            for idx, sub_batch in enumerate(range(batches_per_q)):

                # compute test accuracy every X steps
                if iterations % args.steps_per_test == 0:
                    if rank == 0:
                        Method.update_full_model(Method.model)
                        test_acc = Method.test_full_model(test_data, test_top1)
                        print("Step %d: Top 1 Test Accuracy %.4f" % (iterations-1, test_acc))
                        recorder.add_testacc(test_acc)
                        test_top1.reset()

                x = tf.sparse.slice(x_batch_train, start=[sub_batch * args.train_bs, 0],
                                    size=[args.train_bs, num_features])
                y = tf.sparse.slice(y_batch_train, start=[sub_batch * args.train_bs, 0],
                                    size=[args.train_bs, num_labels])

                # don't count LSH time towards subsequent batches of the mega batch
                if idx == 1:
                    lsh_time = 0

                init_time = time.time()

                # communicate models amongst devices (if multiple devices are present)
                if size > 1:
                    model, comm_time = Method.communicate(Method.model, smart=smartavg)

                # transform sparse label to dense sub-label
                batch = x.get_shape()[0]
                # y_true = get_partial_label(y, cur_idx, batch, num_labels)
                y_true, leftover, label_frac = get_partial_label_mask(y, cur_idx, per_sample_idx, batch, args, idx)

                #t = time.time()
                # perform gradient update
                with tf.GradientTape() as tape:
                    y_pred = Method.model(x)
                    # custom loss
                    max_logit = tf.math.maximum(tf.math.reduce_max(y_pred, axis=1, keepdims=True), 0)
                    # inner exponential sum
                    inner_exp_sum = tf.reduce_sum(tf.math.exp(y_pred - max_logit), axis=1, keepdims=True)
                    # outer exponential sum
                    outside_e_logit = tf.math.maximum(tf.math.exp(-max_logit), 1e-12)
                    outer_exp_sum = num_diff * outside_e_logit
                    # sum of inner and outer
                    e_sum = inner_exp_sum + outer_exp_sum
                    # log of inner and outer sum
                    log_sum_exp = tf.math.log(e_sum)
                    log_sm = y_pred - max_logit - log_sum_exp
                    # inner loss (using mask)
                    inner = tf.reduce_sum(tf.math.multiply(log_sm, y_true), axis=1, keepdims=True)
                    # outer loss
                    # outer = leftover * label_frac * tf.math.log(outside_e_logit / e_sum)
                    outer = leftover * label_frac * (tf.math.log(outside_e_logit) - log_sum_exp)
                    loss_value = -tf.reduce_mean(inner + outer)

                grads = tape.gradient(loss_value, Method.model.trainable_weights)
                optimizer.apply_gradients(zip(grads, Method.model.trainable_weights))
                #print(time.time()-t)
                #print('====')

                # compute accuracy (top 1) and loss for the minibatch
                rec_init = time.time()
                losses.update(np.array(loss_value), batch)
                acc1 = compute_accuracy_lsh(y_pred, y, cur_idx, num_labels)
                top1.update(acc1, batch)
                record_time = time.time() - rec_init
                comp_time = (time.time() - init_time) - (record_time + comm_time)

                if sub_batch != 0:
                    lsh_time = 0

                # store and save accuracy and loss values
                recorder.add_new(comp_time + comm_time + lsh_time, comp_time, comm_time, lsh_time, acc1, test_acc,
                                 loss_value.numpy(), top1.avg, losses.avg)
                recorder.save_to_file()

                # log every X batches
                total_batches += batch
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
