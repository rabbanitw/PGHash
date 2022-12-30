import numpy as np
import tensorflow as tf
import time
from misc import compute_accuracy_lsh
from mpi4py import MPI

if __name__ == '__main__':
    print('test')

    # mpi info
    rank = MPI.COMM_WORLD.Get_rank()
    size = MPI.COMM_WORLD.Get_size()

    
