import torch
import torch.optim as optim
import numpy as np
import gym
from gym.spaces import Discrete, Box
import pickle
import os
from VPG.policy import DiscretePolicy, ContinuousPolicy

class VPG:
    def __init__(self, env):
        self.env_name = env
        self.env = gym.make(env)
        self.observation_dim = self.env.observation_space.shape[0]

        action_space = self.env.action_space
        self.discrete = isinstance(action_space, Discrete)

        if self.discrete:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]

        self.policy = self.build_model()
        # for name, param in self.policy.named_parameters():
        #     if param.requires_grad:
        #         print(name, param.data)

    def build_model(self, hidden_layers=[32]):
        if self.discrete:
            return DiscretePolicy(self.observation_dim, hidden_layers, self.action_dim)
        else:
            return ContinuousPolicy(self.observation_dim, hidden_layers, self.action_dim)

    def save_checkpoint(self):
        torch.save(self.policy.state_dict(), "./weights/" + self.env_name)

    def load_checkpoint(self):
        self.policy.load_state_dict(torch.load("./weights/" + self.env_name))

    def train(self, batch_size=5000, epochs=100, lr=1e-2, render=False):
        return_means, return_stds = [], []
        for epoch in range(epochs):
            loss, returns, lens = self.train_step(batch_size, lr, render=render)
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

    def get_advantage(self, batch_rewards, batch_returns):
        return torch.Tensor(batch_rewards)

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
                batch_rewards += [total_reward] * num_steps
                # batch_rewards += list(self.reward_to_go(ep_rewards))

                observation, ep_rewards, done = env.reset(), [], False
                if len(batch_observations) > batch_size:
                    break

        # 2. Policy Gradient Update Step
        _, log_probs = self.policy(torch.Tensor(batch_observations),
                                   torch.Tensor(batch_actions))
        # advantages = torch.Tensor(batch_rewards)
        advantages = self.get_advantage(batch_rewards, batch_returns)

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

        env.close()
        print('avg return: %.3f \t avg ep_len: %.3f' %
              (np.mean(batch_returns), np.mean(batch_lens)))

class VPGWithAverageBaseline(VPG):
    def __init__(self, env):
        super(VPGWithAverageBaseline, self).__init__(env)

    def get_advantage(self, batch_rewards, batch_returns):
        avg = np.mean(batch_returns)
        adv = np.array(batch_rewards) - avg
        print(avg, adv)
        return torch.Tensor(adv)

if __name__ == "__main__":
    vpg = VPG("CartPole-v0")
    train = True
    if train:
        vpg.train(render=False)
    else:
        vpg.load_checkpoint()
        vpg.evaluate(render=True)