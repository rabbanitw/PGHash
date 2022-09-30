import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torchmetrics
import os
from mlp import NeuralNetwork



def train():
    # Set fixed random number seed
    torch.manual_seed(42)

    # Prepare CIFAR-10 dataset
    dataset = CIFAR10(os.getcwd(), download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

    # Initialize the MLP
    # input_dims = [32*32*3]
    # output_dims = [10]
    input_dims = [32*32*3, 64, 32]
    output_dims = [64, 32, 10]
    mlp = NeuralNetwork(input_dims, output_dims)

    # Define the loss function and optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-4)
    train_accuracy = torchmetrics.Accuracy()

    # Run the training loop
    for epoch in range(0, 5):  # 5 epochs at maximum

        # Print epoch
        print(f'Starting epoch {epoch + 1}')

        # Set current loss value
        current_loss = 0.0

        # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):

            # Get inputs
            inputs, targets = data

            # Zero the gradients
            optimizer.zero_grad()

            # Perform forward pass
            outputs = mlp(inputs)

            # Compute loss
            loss = loss_function(outputs, targets)

            # Compute training accuracy
            train_accuracy(outputs, targets)

            # Perform backward pass
            loss.backward()

            # Perform optimization
            optimizer.step()

            # Print statistics
            current_loss += loss.item()
            if i % 500 == 499:
                print('Loss after mini-batch %5d: %.3f' %
                      (i + 1, current_loss / 500))
                current_loss = 0.0

        print('Epoch %d Accuracy: %f' % (epoch+1, train_accuracy.compute()))
        train_accuracy.reset()

    # Process is complete.
    print('Training process has finished.')


if __name__ == '__main__':
    train()
