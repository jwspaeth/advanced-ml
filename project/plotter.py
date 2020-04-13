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
    losses = [trial_results["losses"] for trial_results in all_trials_results]
    losses = np.stack(losses, axis=0)
    print("Number of rewards: {}".format(rewards.shape))
    print("Number of losses: {}".format(losses.shape))

    average_rewards = np.mean(rewards, axis=0)
    average_losses = np.mean(losses, axis=0)

    # Create fig
    fig, axs = plt.subplots(1, 2)

    # Plot rewards
    for i in range(rewards.shape[0]):
        axs[0].plot(rewards[i], color="b", alpha=.5)
    
    # Plot average reward
    avg = round(np.mean(average_rewards[len(average_rewards)-100-1:]), 2)
    axs[0].plot(average_rewards, color="r", alpha=.7, label="100 Ep. Avg.: {}".format(avg))
    axs[0].set_title("Rewards")
    axs[0].set_ylim([-50, 550])
    axs[0].legend()

    # Plot losses
    for i in range(losses.shape[0]):
        axs[1].plot(losses[i], color="b", alpha=.5)

    # Plot average losses
    axs[1].plot(average_losses, color="r", alpha=.7)
    axs[1].set_title("Losses")
    axs[1].legend()

    plt.show()

if __name__ == "__main__":
    plot_agent(sys.argv[1])