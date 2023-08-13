import torch
import math
import torch.nn.functional as F


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

    def hidden_forward(self, x):
        x = F.relu(self.fc1(x))
        return x
