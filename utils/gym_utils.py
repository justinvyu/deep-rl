import gym
from gym.spaces import Discrete, Box

def get_observation_dim(env):
    """Takes in the actual environment object after `gym.make`"""
    return env.observation_space.shape[0]

def get_action_dim(env):
    if is_discrete(env):
        return env.action_space.n
    return env.action_space.shape[0]

def is_discrete(env):
    action_space = env.action_space
    return isinstance(action_space, Discrete)