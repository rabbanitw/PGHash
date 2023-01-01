import numpy as np


def flatten_weights(weight_list):
    return np.concatenate(list(weight_list[i].flatten() for i in range(len(weight_list))))


def unflatten_weights(flat_weights, layer_shapes, layer_sizes):
    unflatten_model = []
    start_idx = 0
    end_idx = 0
    for i in range(len(layer_shapes)):
        layer_size = layer_sizes[i]
        end_idx += layer_size
        unflatten_model.append(flat_weights[start_idx:end_idx].reshape(layer_shapes[i]))
        start_idx += layer_size
    return unflatten_model


def get_model_architecture(model):
    # find shape and total elements for each layer of the resnet model
    model_weights = model.get_weights()
    layer_shapes = []
    layer_sizes = []
    for i in range(len(model_weights)):
        layer_shapes.append(model_weights[i].shape)
        layer_sizes.append(model_weights[i].size)
    return layer_shapes, layer_sizes


def update_full_model(full_model, weights, biases, idx, start_idx_b, num_features, hidden_layer_size):
    start_idx_w = ((num_features * hidden_layer_size) + hidden_layer_size + (4 * hidden_layer_size))
    bias_idx = idx + start_idx_b
    full_idx = [range((start_idx_w + i*hidden_layer_size), (start_idx_w + i*hidden_layer_size + hidden_layer_size))
                for i in idx]
    full_model[full_idx] = weights.T
    full_model[bias_idx] = biases
    return full_model


def get_sub_model(full_model, idx, start_idx_b, num_features, hidden_layer_size):
    start_idx_w = ((num_features * hidden_layer_size) + hidden_layer_size + (4 * hidden_layer_size))
    # get biases
    bias_idx = idx + start_idx_b
    biases = full_model[bias_idx]
    # get weights
    full_idx = [range((start_idx_w + i*hidden_layer_size), (start_idx_w + i*hidden_layer_size + hidden_layer_size))
                for i in idx]
    weights = full_model[full_idx].T
    return weights, biases


def get_full_dense(full_model, num_features, num_labels, hidden_layer_size):
    dense_shape = (num_labels, hidden_layer_size)
    start_idx = ((num_features * hidden_layer_size) + hidden_layer_size + (4 * hidden_layer_size))
    end_idx = ((num_features * hidden_layer_size) + hidden_layer_size + (4 * hidden_layer_size) +
               hidden_layer_size*num_labels)
    return full_model[start_idx:end_idx].reshape(dense_shape).T


def get_partial_model(full_model, layer_shapes, layer_sizes, num_features, hidden_layer_size):
    end_idx = ((num_features * hidden_layer_size) + hidden_layer_size + (4 * hidden_layer_size))
    return unflatten_weights(full_model[:end_idx], layer_shapes, layer_sizes)
