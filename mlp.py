import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_dims, output_dims):
        self.input_dims = input_dims
        self.output_dims = output_dims
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        # construct MLP
        modules = []
        for i in range(len(input_dims)):
            modules.append(nn.Linear(self.input_dims[i], self.output_dims[i]))
            modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*modules)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.mlp(x)
        return logits
