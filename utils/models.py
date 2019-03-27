import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=F.relu, output_activation=None):
        super(MLP, self).__init__()
        assert len(layer_sizes) > 0, "Must have at least one hidden layer"
        self.activation = activation
        self.output_activation = output_activation
        layers = nn.ModuleList()
        # print(input_size, layer_sizes, output_size)
        layers.append(nn.Linear(input_size, layer_sizes[0]))
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        layers.append(nn.Linear(layer_sizes[len(layer_sizes) - 1], output_size))
        self.layers = layers

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        if self.output_activation:
            x = self.output_activation(self.layers[:-1](x))
        else:
            x = self.layers[-1](x)
        return x