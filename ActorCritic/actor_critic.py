from VPG.vpg import VPG
from utils.models import MLP
import torch.optim as optim
import torch
import torch.nn as nn
import numpy as np

class ActorCritic(VPG):
    def __init__(self, env):
        super(ActorCritic, self).__init__(env)

        # Rename things for fun.
        self.actor = self.policy

        # Critic is a function that maps states to the expected value of the state.
        self.critic = MLP(input_size=self.observation_dim,
                          layer_sizes=[32, 32], output_size=1)

    def train_step(self, batch_size, lr=1e-2, critic_train_epochs=500, render=False):
        """
        1. Sample policy to collect trajectories/experience.
        2. Fit an approximate value function with the sampled rewards.
        3. Evaluate advantages A = r(s, a) + V(s') - V(s)
        4. Approximate the gradient w.r.t the model parameters using a sampled estimate.
            (Using `batch_size` number of samples.)
        5. Take one step of gradient descent on the policy using the gradient.
        """
        env = self.env

        # Set up optimizers.
        actor_optim = optim.Adam(self.actor.parameters(), lr=lr)
        critic_optim = optim.Adam(self.critic.parameters(), lr=lr)

        # 1. Sample policy.
        obs, acts, rewards, returns, lens = self.sample_policy(batch_size, render)

        # 2. Fit critic (value function).
        # Try using the reward-to-go as the targets first. Try bootstrapping later.
        # X = obs, y = rewards
        X = torch.Tensor(obs)
        y = torch.Tensor(rewards)

        criterion = nn.MSELoss() # Least squares regression cost
        for epoch in range(critic_train_epochs + 1):
            pred_value = self.critic(X)

            critic_optim.zero_grad()
            loss = criterion(pred_value, y)
            loss.backward()
            critic_optim.step()

            if epoch % 50 == 0:
                print("epoch: {0} \t loss: {1}".format(epoch, loss.data[0]))

        # 3. Evaluate advantages.
        future_vals = np.array(self.critic(torch.Tensor(obs[1:])))
        future_vals.append(0)
        # Filter out last state values. They should be 0, since there is no
        # future reward in those terminating states.
        last_states = np.array(lens) - 1
        future_vals[filter] = 0

        advantages = rewards + future_vals - self.critic(obs)

        # 4. Approximate policy gradient.
        _, log_probs = self.actor(torch.Tensor(obs),
                                  torch.Tensor(acts))
        actor_optim.zero_grad()
        surrogate_loss = -(log_probs * advantages).mean()
        surrogate_loss.backward()

        # 5. Take one step of gradient descent.
        actor_optim.step()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    args = parser.parse_args()

    ac = ActorCritic(args.env)