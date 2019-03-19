import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
import gym
import load_policy
import os
import matplotlib.pyplot as plt
import tf_util
from run_trained_policy import run_policy

def main(env_name, train=True):
    """
    DAgger: Dataset Aggregation
    Goal: collect data from the learned policy instead of the true policy of the data.
    Solution: DAgger solves the distributional drift problem by making the policy of the data
    converge to the optimal learned policy.

    while not converged:
        1. Train pi_theta(a_t | o_t) from dataset D={o_1, a_1, ..., o_n, a_n}
        2. Run pi_theta(a_t | o_t) to get dataset D_pi={o_1, ..., o_m}
        3. Label D_pi with actions a_t, given by the expert policy.
        4. Aggregate D <- D + D_pi
    """
    # env_name = "HalfCheetah-v2"
    env_name = "Humanoid-v2"
    with open("expert_data/" + env_name + ".pkl", "rb") as f:
        data = pickle.load(f)

    observations = np.array(data['observations'])
    N = len(observations)
    actions = np.array(data['actions'])
    action_dim = actions.shape[1:][1]
    actions = actions.reshape((N, action_dim))
    observation_shape = observations.shape[1:]

    model = tf.keras.Sequential([
        keras.layers.Dense(128, activation="relu", input_shape=observation_shape),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(action_dim)  # Output a vector of the action dimension
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),
                  loss='mse')
    # model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
    #               loss='mse')

    D, test_X, D_labels, test_y = train_test_split(observations, actions, test_size=0.5)

    if train:
        with tf.Session():
            tf_util.initialize()

            # epochs = 50
            epochs = 25
            convergence_threshold = 0.0001
            std_loss = 1

            loss_history = []
            mean_history = []
            std_history = []
            dataset_size_history = []
            dagger_cycles = 0
            while dagger_cycles < 50 or std_loss > convergence_threshold:
                print("D size:", D.shape)
                print("D_labels size:", D_labels.shape)

                print("Sampling policy and labeling with expert data to build D_pi")
                D_pi, D_pi_labels, returns = sample_policy(model, env_name)

                D = np.vstack((D, D_pi))
                D_labels = np.vstack((D_labels, D_pi_labels))

                mean_history.append(np.mean(returns))
                std_history.append(np.std(returns))
                print(returns, mean_history, std_history)
                dataset_size_history.append(len(D))
                dagger_cycles += 1

                results = {"mean": np.array(mean_history),
                           "std": np.array(std_history),
                           "size": np.array(dataset_size_history),
                           "num_iters": dagger_cycles,
                           "loss": np.array(loss_history)}
                with open(os.path.join('dagger_history', env_name + '.pkl'), 'wb') as f:
                    pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)

                # Fit model on new aggregated dataset
                history = model.fit(D, D_labels, epochs=epochs, batch_size=32)
                loss = history.history["loss"]
                std_loss = np.std(loss)

                # Save model bc_weights and loss history
                model.save_weights("./dagger_weights/" + env_name)
                loss_history += loss
                print(std_loss)

    else:
        model.load_weights('./dagger_weights/' + env_name)
        run_policy(env_name, model, render=True)

def sample_policy(policy, env_name):
    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit
    # num_rollouts = 5
    num_rollouts = 25
    render = False

    expert_policy = load_policy.load_policy("./experts/" + env_name + ".pkl")
    observations = []
    expert_actions = []
    returns = []

    for i in range(num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy.predict(obs[None, :])
            expert_action = expert_policy(obs[None, :])[0] # Flatten from 2D
            observations.append(obs)
            expert_actions.append(expert_action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if steps >= max_steps:
                break
        returns.append(totalr)
    return np.array(observations), np.array(expert_actions), np.array(returns)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("env", type=str)
    parser.add_argument("--train", type=bool, default=False)
    args = parser.parse_args()

    if args.train:
        main(args.env)
    else:
        main(args.env, train=False)