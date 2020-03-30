#!/usr/bin/env python3

import sys
import glob
import pickle

import numpy as np
import matplotlib.pyplot as plt

def plot_agent(agent_folder):
    # Plot results
    print("Agent folder: {}".format(agent_folder))

    trial_folders = glob.glob("{}/*".format(agent_folder))
    print("Trial folders: {}".format(trial_folders))
    all_trials_results = []
    for trial_folder in trial_folders:
        with open("{}/results_dict.pkl".format(trial_folder), "rb") as f:
            all_trials_results.append(pickle.load(f))

    rewards = [trial_results["rewards"] for trial_results in all_trials_results]
    rewards = np.stack(rewards, axis=0)
    avgs = np.mean(rewards[:, rewards.shape[1]-100-1:], axis=1)
    print("Rewards shape: {}".format(rewards.shape))
    print("Averages shape: {}".format(avgs.shape))
    print("Averages: {}".format(avgs))

    fig, axs = plt.subplots(1)
    for i in range(rewards.shape[0]):
        axs.plot(rewards[i])

    axs.set_title("Performance")
    axs.set_xlabel("Episode")
    axs.set_ylabel("Total Reward")
    axs.set_ylim([-550, 50])
    axs.legend()

    plt.show()

if __name__ == "__main__":
    plot_agent(sys.argv[1])