import numpy as np
import argparse
from dataloader import load_extreme_data
from mlp import Net
from mpi4py import MPI
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
from adam import Adam
import time
cudnn.benchmark = True
torch.set_default_dtype(torch.float32)

# from network import Graph
# from communicators import CentralizedSGD, LSHCentralizedSGD
# from unpack import get_model_architecture, flatten_tensors


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    # add arguments
    parser.add_argument('--name', type=str, default='Test')
    parser.add_argument('--dataset', type=str, default='Delicious200K')
    parser.add_argument('--graph_type', type=str, default='fully_connected')
    parser.add_argument('--hash_type', type=str, default='pghash')
    parser.add_argument('--dwta', type=int, default=0)
    parser.add_argument('--randomSeed', type=int, default=1203)
    parser.add_argument('--sdim', type=int, default=8)
    parser.add_argument('--c', type=int, default=8)
    parser.add_argument('--num_tables', type=int, default=50)
    parser.add_argument('--lr', type=int, default=1e-4)
    parser.add_argument('--train_bs', type=int, default=200)
    parser.add_argument('--test_bs', type=int, default=4096)
    parser.add_argument('--steps_per_lsh', type=int, default=50)
    parser.add_argument('--steps_per_test', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--hidden_layer_size', type=int, default=128)
    parser.add_argument('--cr', type=float, default=1)

    # parse the argument
    args = parser.parse_args()

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    # set random seed
    torch.manual_seed(args.randomSeed)
    np.random.seed(args.randomSeed)
    # set default memory size
    torch.set_default_dtype(torch.float32)

    # hashing parameters
    sdim = args.sdim
    num_tables = args.num_tables
    hash_type = args.hash_type
    steps_per_lsh = args.steps_per_lsh

    # load base network topology
    graph_type = args.graph_type
    weight_type = None
    num_clusters = None
    # G = Graph(rank, size, MPI.COMM_WORLD, graph_type, weight_type, num_c=num_clusters)

    # training parameters
    epochs = args.epochs
    hls = args.hidden_layer_size
    train_data_path = '../data/' + args.dataset + '/train.txt'
    test_data_path = '../data/' + args.dataset + '/test.txt'

    print('Loading and partitioning data...')
    train_data, tf_data, n_features, n_labels = load_extreme_data(rank, size, args.train_bs, train_data_path,
                                                                  test_data_path)
    test_data = None

    print('Initializing model...')

    model = Net(n_features, args.hidden_layer_size, n_labels, args.c, num_tables, sdim, args.cr, steps_per_lsh)
    optimizer = Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x, y) in enumerate(train_data):
            t = time.time()
            loss = model(x, y)
            # print(time.time()-t)
            print(loss)
            loss.backward()
            # print(time.time() - t)
            optimizer.step()
            optimizer.zero_grad()
            # print(time.time()-t)
            if step % 100 == 0:
                print('Loss is %f' % loss)

