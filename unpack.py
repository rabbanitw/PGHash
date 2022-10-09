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


def layer_compression(model, max_idx):
    weights = model.get_weights()
    final_dense = weights[-2]
    new_final_dense = np.zeros_like(final_dense)
    new_final_dense[max_idx] = final_dense[max_idx]
    weights[-2] = new_final_dense
    model.set_weights(weights)


