import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np
import gym
import pickle
import os

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

class CategoricalPolicy(nn.Module):
    def __init__(self, observation_dim, hidden_layers, action_dim, activation=F.relu):
        super(CategoricalPolicy, self).__init__()

        # Represent the function that calculates action logits as an MLP.
        self.logits = MLP(observation_dim, hidden_layers, action_dim, activation=activation)
        self.model_representation = [observation_dim] + hidden_layers + [action_dim]

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
        self.env_name = env
        self.env = gym.make(env)
        self.observation_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.policy = self.build_model()

    def build_model(self, hidden_layers=[32]):
        return CategoricalPolicy(self.observation_dim, hidden_layers, self.action_dim)

    def save_checkpoint(self):
        torch.save(self.policy.state_dict(), "./weights/" + self.env_name)

    def load_checkpoint(self):
        self.policy.load_state_dict(torch.load("./weights/" + self.env_name))

    def train(self, batch_size=5000, epochs=100, lr=1e-2):
        return_means, return_stds = [], []
        for epoch in range(epochs):
            loss, returns, lens = self.train_step(batch_size, lr)
            print('epoch: %3d \t loss: %.3f \t avg return: %.3f \t ep_len: %.3f' %
                  (epoch, loss, np.mean(returns), np.mean(lens)))
            return_means.append(np.mean(returns))
            return_stds.append(np.std(returns))
            self.save_checkpoint()

        with open(os.path.join('train_history', self.env_name + '.pkl'), 'wb') as f:
            pickle.dump({"env": self.env_name,
                         "return_means": np.array(return_means),
                         "return_stds": np.array(return_stds),
                         "batch_size": batch_size,
                         "lr": lr,
                         "policy_layer_representation": self.policy.model_representation},
                        f, pickle.HIGHEST_PROTOCOL)

    def reward_to_go(self, rewards):
        """
        Calculates the total reward collected from the current time-step t
        until the end of the episode. Reduces variance, using the intuition that
        present cannot affect the past.

        >>> list(VPG.reward_to_go(None, [-1, 0, 1, 2, 3]))
        [5.0, 6.0, 6.0, 5.0, 3.0]
        """
        n = len(rewards)
        rewards_to_go = np.zeros(n)
        for i in reversed(range(n)):
            rewards_to_go[i] = rewards[i] + (rewards_to_go[i + 1] if i + 1 < n else 0)
        return rewards_to_go

    def train_step(self, batch_size, lr=1e-2, render=False):
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

        batch_returns = []
        batch_lens = []

        observation = env.reset()
        while True:
            batch_observations.append(observation.copy())

            # Sample action from policy.
            action, _ = self.policy(torch.Tensor(observation))
            observation, reward, done, _ = env.step(action.numpy())
            batch_actions.append(action)
            ep_rewards.append(reward)
            if render:
                env.render()

            # Handle what happens when the episode is over.
            if done:
                total_reward, num_steps = sum(ep_rewards), len(ep_rewards)
                batch_returns.append(total_reward)
                batch_lens.append(num_steps)

                # Each step in this trajectory is associated with the end reward.
                batch_rewards += list(self.reward_to_go(ep_rewards))

                observation, ep_rewards, done = env.reset(), [], False
                if len(batch_observations) > batch_size:
                    break

        # 2. Policy Gradient Update Step
        _, log_probs = self.policy(torch.Tensor(batch_observations),
                                   torch.Tensor(batch_actions))
        advantages = torch.Tensor(batch_rewards) # Most basic implementation.

        # Define loss function.
        surrogate_loss = -(log_probs * advantages).mean()
        # Take one step of gradient descent.
        optimizer.zero_grad()
        surrogate_loss.backward()
        optimizer.step()

        return surrogate_loss, batch_returns, batch_lens

    def evaluate(self, num_rollouts=25, render=False):
        env = self.env

        batch_returns = []
        batch_lens = []
        for i in range(num_rollouts):
            observation = env.reset()
            ep_rewards = []
            done = False
            while not done:
                action, _ = self.policy(torch.Tensor(observation))
                observation, reward, done, _ = env.step(action.numpy())
                ep_rewards.append(reward)
                if render:
                    env.render()

            ep_return, ep_len = sum(ep_rewards), len(ep_rewards)
            batch_returns.append(ep_return)
            batch_lens.append(ep_len)

            print('rollout #: %3d \t return: %.3f \t ep_len: %.3f' %
                  (i, ep_return, ep_len))

        print('avg return: %.3f \t avg ep_len: %.3f' %
              (np.mean(batch_returns), np.mean(batch_lens)))

if __name__ == "__main__":
    vpg = VPG("CartPole-v0")
    train = True
    if train:
        vpg.train()
    else:
        vpg.load_checkpoint()
        vpg.evaluate()