import torch
from torch.utils.data import Dataset, DataLoader
import random


class SparseDataset(Dataset):
    """
    Custom Dataset class for scipy sparse matrix
    """

    def __init__(self, data, targets, coo=True):

        if coo:
            self.data = data.tocsr()
            self.targets = targets.tocsr()
        else:
            self.data = data
            self.targets = targets

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return self.data.shape[0]


def data_generator_csr(file, batch_size, num_features, num_classes):
    with open(file, 'r', encoding='utf-8') as f:
        f.readline()  # ignore the header
        lines = f.readlines()
        num_data = len(lines)
        count = 0

        # sparse data lists
        crow_indices_data = [0]
        col_indices_data = []
        vals_data = []

        # sparse label lists
        crow_indices_label = [0]
        col_indices_label = []
        vals_label = []

        # randomize data
        random.shuffle(lines)

        for line in lines:

            # parse line
            itms = line.strip().split(' ')
            labels_idxs = [int(itm) for itm in itms[0].split(',')]
            num_labels = len(labels_idxs)

            # add the indices for each line for each non-zero label
            col_indices_label += labels_idxs

            # add number of non-zero elements per line/row
            crow_indices_label.append(num_labels + crow_indices_label[-1])
            crow_indices_data.append(len(itms[1:]) + crow_indices_data[-1])

            # add label values
            vals_label += ([1.0 / num_labels] * num_labels)

            # iterate to find indices for each line for each non-zero for data and value for data
            for itm in itms[1:]:
                col_indices_data.append(int(itm.split(':')[0]))
                vals_data.append(float(itm.split(':')[1]))
            count += 1

    # turn into a sparse csr tensor
    sparse_label = torch.sparse_csr_tensor(torch.tensor(crow_indices_label, dtype=torch.int64),
                                           torch.tensor(col_indices_label, dtype=torch.int64),
                                           torch.tensor(vals_label, dtype=torch.float32), size=(num_data, num_classes))

    sparse_data = torch.sparse_csr_tensor(torch.tensor(crow_indices_data, dtype=torch.int64),
                                          torch.tensor(col_indices_data, dtype=torch.int64),
                                          torch.tensor(vals_data, dtype=torch.float32), size=(num_data, num_features))

    # load and batch data
    sparse_dataset = SparseDataset(sparse_data, sparse_label, coo=False)
    dataloader = DataLoader(sparse_dataset, batch_size=batch_size, shuffle=True)

    return dataloader
