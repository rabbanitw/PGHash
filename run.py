import tensorflow as tf
from dataloader import load_amazon670
from train import train
from dsgd import DecentralizedSGD


if __name__ == '__main__':
    batch_size = 64
    epochs = 2
    print('Loading data...')
    train_data, test_data = load_amazon670(batch_size)
    print('Beginning training...')
    train(train_data, test_data, epochs)
