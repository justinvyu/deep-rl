import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

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

class DiscretePolicy(nn.Module):
    def __init__(self, observation_dim, hidden_layers, action_dim, activation=F.relu):
        super(DiscretePolicy, self).__init__()

        # Represent the function that calculates action logits as an MLP.
        self.logits = MLP(observation_dim, hidden_layers, action_dim, activation=activation)
        self.model_representation = [observation_dim] + hidden_layers + [action_dim]

    def forward(self, x, actions=None):
        observation = x
        # Get the log probabilities of actions, given the observation.
        logits = self.logits(observation)
        # Now, create a categorical distribution (which is the policy for a discrete-action setting).
        policy = Categorical(logits=logits)
        # Sample from the policy, unless we are provided an action. In that case,
        # calculate the probability of pi(a|s).
        if actions is None:
            sample_act = policy.sample()
        else:
            sample_act = actions
        # Return log probability (for use in creating the surrogate loss function.
        return sample_act, policy.log_prob(sample_act)

class ContinuousPolicy(nn.Module):
    def __init__(self, observation_dim, hidden_layers, action_dim, activation=F.relu):
        super(ContinuousPolicy, self).__init__()

        # Represent the means of each action dimension as an MLP (that estimates a Gaussian model).
        self.mu = MLP(observation_dim, hidden_layers, action_dim, activation=activation)
        # Standard deviations are learned as well.
        self.sigma = nn.Parameter(torch.ones(action_dim))

    def forward(self, x, actions=None):
        observation = x
        # Get the means of the actions.
        means = self.mu(x)
        policy = Normal(means, self.sigma)
        # Sample an action using the Gaussian over each action dimension.
        if actions is None:
            sample_act = policy.sample()
        else:
            sample_act = actions
        return sample_act, policy.log_prob(sample_act)

