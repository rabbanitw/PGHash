import tensorflow as tf
import numpy as np


def compute_accuracy(y_true, y_pred, topk=1):
    result = tf.math.top_k(y_pred, k=topk)
    r, c = y_pred.get_shape()
    true_idx = y_true.indices.numpy()
    count = 0
    for i in range(r):
        for j in range(topk):
            top_idx = np.array([i, result.indices[i, j].numpy()])
            count += int(np.any(np.all(top_idx == true_idx, axis=1)))
    return count/(r*topk)
