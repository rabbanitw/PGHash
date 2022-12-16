import numpy as np
import torch
from train import accuracy

if __name__ == '__main__':
    a = torch.rand(3, 6)
    b = torch.eye(10)
    print(a)
    # print(b)
    # acc = accuracy(a, b)
    print(a.sum(dim=-1).unsqueeze(-1))
    a = a / a.sum(dim=-1).unsqueeze(-1)
    print(a)

