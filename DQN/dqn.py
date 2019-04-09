import numpy as np
import random
from utils.models import MLP
from utils import gym_utils
import gym

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    @property
    def size(self):
        return len(self.memory)

    def flush(self):
        """
        Flushes the replay buffer of old experiences if the capacity is reached.
        >>> b = ReplayBuffer(5)
        >>> b.memory = [1, 2, 3, 4, 5, 6]
        >>> b.flush()
        >>> b.memory
        [2, 3, 4, 5, 6]
        """
        assert self.size >= self.capacity
        self.memory = self.memory[self.size - self.capacity:self.size]

    def add(self, experiences):
        """
        Adds a list of experiences to the replay buffer.
        >>> b = ReplayBuffer(5)
        >>> b.add([1, 2, 3, 4, 5, 6])
        >>> b.memory
        [2, 3, 4, 5, 6]
        """
        self.memory.extend(experiences)
        if self.size >= self.capacity:
            self.flush()

    def sample(self, n):
        """Samples `n` items from the buffer uniformly."""
        assert n <= self.capacity, "Number of sampled items needs to be less than the buffer capacity"
        return random.sample(self.memory, n)

class DQN:
    def __init__(self,
                 env_name,
                 q_func,
                 target_q_func,
                 replay_buffer_capacity=1000000,
                 batch_size=32,
                 gamma=0.99,
                 learning_starts=50000,
                 learning_freq=4,
                 target_update_freq=10000,
                 double_q=True):
        # Set up environment.
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Parameters of DQN
        self.replay_buffer_capacity = replay_buffer_capacity
        self.batch_size = batch_size
        self.gamma = gamma
        self.learning_starts = learning_starts
        self.learning_freq = learning_freq
        self.target_update_freq = target_update_freq
        self.double_q = double_q

        self.last_obs = self.env.reset()
        self.t = 0
        self.num_param_updates = 0

        # Create Q-network/target network.
        self.observation_dim, self.action_dim = gym_utils.get_observation_dim(self.env), \
                                                gym_utils.get_action_dim(self.env)
        # Build model to perform regression for the Q-value (single real number output)
        self.q_func = q_func
        # Parameters will be copied from `self.Q` into the target network.
        self.target_q_func = target_q_func

        # Create replay buffer.
        self.memory = ReplayBuffer(replay_buffer_capacity)

    def train(self, num_timesteps=1e8):
        """
        Training steps:
        1. Collect experience, add to replay buffer.
        2. Sample mini-batch from replay buffer.
        3. Compute y_j = r_j + gamma * max_a' Q'(s',a') <- using target network.
        4. K epochs of regression, updating parameters phi of `self.Q`
        5. Update target network Q'.
        """
        while self.t < num_timesteps:
            self.env_step()
            self.update_model()

    def env_step(self):
        # ...

        self.t += 1

    def update_model(self):
        if self.t % self.learning_freq == 0 and self.memory.size >= self.learning_starts:
            # ...
            self.num_param_updates += 1

if __name__ == "__main__":
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    obs_dim = gym_utils.get_observation_dim(env)
    act_dim = gym_utils.get_action_dim(env)
    q_func = MLP(obs_dim, [32, 32], act_dim)
    target = MLP(obs_dim, [32, 32], act_dim)

    dqn = DQN(
        env_name=env_name,
        q_func=q_func,
        target_q_func=target,
    )

