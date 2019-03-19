import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import os

env = "Humanoid-v2"

data_path = "./dagger_history/" + env + ".pkl"
expert_data_path = "./expert_return_data/" + env + ".pkl"

with open(data_path, "rb") as f:
    data = pickle.load(f)

with open(expert_data_path, "rb") as f:
    expert_data = pickle.load(f)

# results = {"mean": np.array(mean_history),
#            "std": np.array(std_history),
#            "size": np.array(dataset_size_history),
#            "num_iters": dagger_cycles,
#            "loss": np.array(loss_history)}

sns.set(style="darkgrid")

means = data["mean"]
stds = data["std"]
dataset_sizes = data["size"]
num_iters = data["num_iters"]
loss = data["loss"]

expert_means = expert_data["mean"]
expert_stds = expert_data["std"]

print(means, stds, dataset_sizes, num_iters, loss)

print(len(means))
plt.errorbar(range(len(means)),
             means,
             yerr=stds,
             capsize=2, elinewidth=1, markeredgewidth=1)
plt.errorbar(range(len(expert_means)),
             expert_means,
             yerr=expert_stds,
             capsize=2, elinewidth=1, markeredgewidth=1, color='r')

plt.title("Learning Curve of Model trained with DAgger of " + env)
plt.xlabel("Number of DAgger iterations\n(Trained with 25 epochs, adding data from 25 rollouts per iteration)")
plt.ylabel("Mean of Returns (from the 25 rollouts)")
plt.legend(["Trained Policy", "Expert Policy"])
plt.show()