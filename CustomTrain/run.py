import numpy as np
from nn import Model
from dataloader import load_extreme_data
import time


def xavier_init(nodes_in, nodes_out):
    limit = np.sqrt(6 / (nodes_in + nodes_out))
    return np.random.uniform(low=-limit, high=limit, size=(nodes_in, nodes_out))


def init_layers(nn_architecture):
    params_values = {}
    for idx, layer in enumerate(nn_architecture):
        layer_idx = idx + 1
        layer_input_size = layer["input_dim"]
        layer_output_size = layer["output_dim"]
        params_values['W' + str(layer_idx)] = xavier_init(layer_output_size, layer_input_size)
        params_values['b' + str(layer_idx)] = np.zeros((layer_output_size, 1))
    return params_values


def main():
    train_bs = 128
    test_bs = 4096
    dataset = 'Delicious200K'
    train_data_path = '../data/' + dataset + '/train.txt'
    test_data_path = '../data/' + dataset + '/test.txt'
    hls = 128
    seed = 1203
    np.random.seed(seed)
    lr = 1e-4

    print('Begin Loading Data...')
    train_data, test_data, n_features, n_labels = load_extreme_data(0, 1, train_bs, test_bs, train_data_path,
                                                                    test_data_path)

    print('Finished Loading Data...')
    w1 = xavier_init(n_features, hls)
    b1 = np.zeros((1, hls))

    w2 = xavier_init(hls, n_labels)
    b2 = np.zeros((1, n_labels))

    nn = Model(w1, b1, w2, b2, lr)

    train_x = train_data[0]
    train_y = train_data[1]
    num_data = train_x.shape[0]
    num_batchs = int(np.ceil(num_data/train_bs))

    #'''
    for i in range(num_batchs):
        x = train_x[i*train_bs:(i+1)*train_bs, :]
        y = train_y[i * train_bs:(i + 1) * train_bs].toarray()
        y = y / np.count_nonzero(y, axis=1, keepdims=True)
        loss, pred = nn.forward(x, targ=y)
        nn.backward()
        nn.update()
        print(loss)
    #'''






if __name__ == '__main__':
    main()

    # test sparse-dense matrix multiplication out