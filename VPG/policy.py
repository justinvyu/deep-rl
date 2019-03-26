import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from utils.models import MLP

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

