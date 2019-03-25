import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
import os

env = "Acrobot-v1"
data_path = "./train_history/" + env + ".pkl"

with open(data_path, "rb") as f:
    data = pickle.load(f)

sns.set(style="darkgrid")

return_means = data["return_means"]
return_stds = data["return_stds"]
batch_size = data["batch_size"]
lr = data["lr"]
model = data["policy_layer_representation"]
epochs = len(return_means)

print(epochs, return_means, return_stds, model)

plt.errorbar(range(epochs),
             return_means,
             yerr=return_stds,
             capsize=2, elinewidth=1, markeredgewidth=1)

plt.suptitle("VPG Learning Curve on environment: " + env, y=0.99)
plt.title("(Params) batch_size={0}, lr={1}, epochs={2}".format(batch_size, lr, epochs)
          + "\n(Error bars: std of returns sampled during each epoch)", fontsize=10)
plt.xlabel("Epoch")
plt.ylabel("Mean of Returns")
plt.legend(["Trained Policy"])
plt.show()