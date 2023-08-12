import torch
import math
import torch.nn.functional as F


class SparseLinear(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        self.init_parameters()

    def init_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return torch.sparse.addmm(self.bias, x, self.weight.t())


class SimpleNN(torch.nn.Module):

    def __init__(self, num_features, hls, num_labels):
        super(SimpleNN, self).__init__()

        self.linear1 = torch.nn.Linear(num_features, hls)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hls, num_labels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x


class SparseNN(torch.nn.Module):

    def __init__(self, num_features, hls, num_labels):
        super(SparseNN, self).__init__()

        self.linear1 = SparseLinear(num_features, hls)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hls, num_labels)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        return self.linear2(x)

    def hidden_forward(self, x):
        return self.activation(self.linear1(x))


def truncated_normal_init_(tensor, mean=0, std=1):
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/15
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)


class Net(torch.nn.Module):
    def __init__(self, feature_dim, hidden_dim, n_classes):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(feature_dim, hidden_dim)
        truncated_normal_init_(self.fc1.weight, std=2.0/math.sqrt(feature_dim+hidden_dim))
        truncated_normal_init_(self.fc1.bias, std=2.0/math.sqrt(feature_dim+hidden_dim))
        self.fc2 = torch.nn.Linear(hidden_dim, n_classes)
        truncated_normal_init_(self.fc2.weight, std=2.0/math.sqrt(hidden_dim+n_classes))
        truncated_normal_init_(self.fc2.bias, std=2.0/math.sqrt(hidden_dim+n_classes))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
