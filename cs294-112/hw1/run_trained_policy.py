import pickle
import numpy as np
import os

def run_policy(env_name, policy, render=False):
    import gym
    env = gym.make(env_name)
    max_steps = env.spec.timestep_limit
    num_rollouts = 20

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
            action = policy.predict(obs[None, :])
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