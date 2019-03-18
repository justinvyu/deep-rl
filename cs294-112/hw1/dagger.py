import tensorflow as tf
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from tensorflow import keras
import os
import matplotlib.pyplot as plt

def main():
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
    pass

if __name__ == "__main__":
    main()