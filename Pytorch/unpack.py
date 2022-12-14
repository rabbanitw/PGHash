import numpy as np
import torch


def flatten_tensors(tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.
    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.
    Arguments:
        tensors (Iterable[Tensor]): dense tensors to flatten.
    Returns:
        A 1D buffer containing input tensors.
    """
    if len(tensors) == 1:
        return tensors[0].view(-1).clone()
    flat = torch.cat([t.view(-1) for t in tensors], dim=0)
    return flat


def unflatten_tensors(flat, tensors):
    """
    Reference: https://github.com/facebookresearch/stochastic_gradient_push
    View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by flatten_dense_tensors.
    Arguments:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
            unflatten flat.
    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    outputs = []
    offset = 0
    for tensor in tensors:
        numel = tensor.numel()
        outputs.append(flat.narrow(0, offset, numel).view_as(tensor))
        offset += numel
    return tuple(outputs)


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
