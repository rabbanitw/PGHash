import numpy as np
import torch
from train import accuracy

if __name__ == '__main__':
    a = torch.rand(10, 10)
    b = torch.eye(10)
    print(a)
    print(b)
    acc = accuracy(a, b)
    print(acc)

