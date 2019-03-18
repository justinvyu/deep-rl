import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import matplotlib.pyplot as plt

def build_model(observation_shape, action_dim):
    # For other envs.
    # model = keras.Sequential([
    #     keras.layers.Dense(64, activation="relu", input_shape=observation_shape),
    #     keras.layers.Dense(32, activation="relu"),
    #     keras.layers.Dense(action_dim) # Output a vector of the action dimension
    # ])
    # model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
    #               loss='mse')
    # For Humanoid-v2
    model = tf.keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=observation_shape),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(action_dim)  # Output a vector of the action dimension
    ])
    model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='mse')
    print(model.summary())
    return model

def get_training_testing_split(env_name):
    with open("expert_data/" + env_name + ".pkl", "rb") as f:
        data = pickle.load(f)

    observations = np.array(data['observations'])
    N = len(observations)
    actions = np.array(data['actions'])
    action_dim = actions.shape[1:][1]
    actions = actions.reshape((N, action_dim))
    # print(actions[:10])

    # 70-30 split for training, testing
    observation_shape = observations.shape[1:]

    print("{} samples".format(N))
    print("Observations are dimension: {}".format(observation_shape))
    print("Actions are dimension: {}".format(action_dim))

    # Split into training/testing sets
    train_X, test_X, train_y, test_y = train_test_split(observations, actions, test_size=0.25)
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # print(train_X[0], train_y[0])
    return train_X, train_y, test_X, test_y

def train_model(env_name, epochs):
    train_X, train_y, _, _ = get_training_testing_split(env_name)
    model = build_model(train_X.shape[1:], train_y.shape[1])
    model.fit(train_X, train_y, epochs=epochs, batch_size=32)
    model.save_weights("./weights/" + env_name)

def evaluate_model(env_name):
    _, _, test_X, test_y = get_training_testing_split(env_name)

    print(test_X.shape, test_y.shape)
    model = build_model(test_X.shape[1:], test_y.shape[1])
    model.load_weights("./weights/" + env_name)
    model.evaluate(test_X, test_y)

def run_policy(env_name):
    _, _, test_X, test_y = get_training_testing_split(env_name)

    import gym
    model = build_model(test_X.shape[1:], test_y.shape[1])
    model.load_weights("./dagger_weights/" + env_name)
    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit
    num_rollouts = 20
    render = False

    returns = []
    observations = []
    actions = []
    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None, :])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                env.render()
            if steps % 100 == 0: print("%i/%i" % (steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)

    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    expert_data = {'observations': np.array(observations),
                   'actions': np.array(actions),
                   'returns': np.array(returns)}

    with open(os.path.join('clone_data', env_name + '.pkl'), 'wb') as f:
        pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--train", type=bool, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    print(args.train, args.epochs)
    if args.train:
        train_model(args.env, args.epochs)
    else:
        # evaluate_model(args.env)
        run_policy(args.env)
