import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import gym

class MLP(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=F.relu, output_activation=None):
        super(MLP, self).__init__()
        assert len(layer_sizes) > 0, "Must have at least one hidden layer"
        self.activation = activation
        self.output_activation = output_activation
        layers = nn.ModuleList()
        print(input_size, layer_sizes, output_size)
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

class CategoricalPolicy(nn.Module):
    def __init__(self, observation_dim, hidden_layers, action_dim, activation=F.relu):
        super(CategoricalPolicy, self).__init__()

        # Represent the function that calculates action logits as an MLP.
        self.logits = MLP(observation_dim, hidden_layers, action_dim, activation=activation)

    def forward(self, x, action=None):
        observation = x
        # Get the log probabilities of actions, given the observation.
        logits = self.logits(observation)
        # Now, create a categorical distribution (which is the policy for a discrete-action setting).
        policy = Categorical(logits=logits)
        # Sample from the policy, unless we are provided an action. In that case,
        # calculate the probability of pi(a|s).
        if action is None:
            sample_act = policy.sample()
        else:
            sample_act = action
        # Return log probability (for use in creating the surrogate loss function.
        return sample_act, policy.log_prob(sample_act)

class VPG:
    def __init__(self, env):
        self.env = gym.make(env)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy = self.build_model()

    def build_model(self, hidden_layers=[32]):
        return CategoricalPolicy(self.observation_dim, hidden_layers, self.action_dim)

    def train(self, batch_size=5000, epochs=50, lr=1e-2):
        """

        :param batch_size:
        :param epochs:
        :param lr:
        :return:
        """
        for epoch in range(epochs):
            self.train_step(batch_size, lr)

    def train_step(self, batch_size, lr=1e-2):
        """
        Performs one epoch of training, which consists of two parts:
        1. Experience Collection
            Experience collection is performed by sampling the current policy to get at least
            `batch_size` samples of (s, a, r). These will be used by the update step.
        2. Policy Gradient Update Step
            The policy gradient update step first creates the surrogate loss function
            L = -mean(log(pi(a|s)) * R(s, a)) over all state, action pairs. Then,
        """
        env = self.env

        # Set up an optimizer on the policy's parameters.
        optimizer = optim.Adam(self.policy.parameters(), lr=lr)

        # 1. Experience Collection
        batch_observations = []
        batch_actions = []
        batch_rewards = []
        ep_rewards = []

        observation = env.reset()
        while True:
            batch_observations.append(observation.copy())

            # Sample action from policy.
            action, _ = self.policy(torch.Tensor(observation))
            observation, reward, done, _ = env.step(action.numpy())
            batch_actions.append(action)
            ep_rewards.append(reward)
            env.render()

            # Handle what happens when the episode is over.
            if done:
                total_reward, num_steps = sum(ep_rewards), len(ep_rewards)
                # Each step in this trajectory is associated with the end reward.
                batch_rewards += [total_reward] * num_steps

                observation, ep_rewards, done = env.reset(), [], False

                if len(batch_observations) > batch_size:
                    break

        # 2. Policy Gradient Update Step
        _, log_probs = self.policy(torch.Tensor(batch_observations),
                                   torch.Tensor(batch_actions))
        advantages = torch.Tensor(batch_rewards) # Most basic implementation.

        # Define loss function.
        surrogate_loss = -(log_probs * advantages).mean()
        optimizer.zero_grad()
        surrogate_loss.backward()
        optimizer.step()

if __name__ == "__main__":
    vpg = VPG("CartPole-v0")
    vpg.train()