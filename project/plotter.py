#!/usr/bin/env python3

import sys
import glob
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt

def plot_agent(agent_folder):
    # Plot results
    print("Agent folder: {}".format(agent_folder))

    trial_folders = glob.glob("{}/*".format(agent_folder))
    print("Trial folders: {}".format(trial_folders))
    all_trials_results = []
    for trial_folder in trial_folders:
        print(os.listdir(trial_folder))
        with open("{}/results_dict.pkl".format(trial_folder), "rb") as f:
            all_trials_results.append(pickle.load(f))

    rewards = [trial_results["rewards"] for trial_results in all_trials_results]
    rewards = np.stack(rewards, axis=0)
    losses = [trial_results["losses"] for trial_results in all_trials_results]
    losses = np.stack(losses, axis=0)
    print("Number of rewards: {}".format(rewards.shape))
    print("Number of losses: {}".format(losses.shape))

    average_rewards = np.mean(rewards, axis=0)
    std_rewards = np.std(rewards, axis=0)
    average_losses = np.mean(losses, axis=0)
    std_losses = np.std(losses, axis=0)

    # Create fig
    fig, axs = plt.subplots(1, 2)
    fig.suptitle("{}".format(agent_folder))
    
    ### Plot average reward ###
    avg = round(np.mean(average_rewards[len(average_rewards)-100-1:]), 2)
    axs[0].plot(average_rewards, color="r", alpha=.7, label="100 Ep. Avg.: {}\nFull avg: {}".format(avg, 
        round(np.mean(average_rewards), 2)))
    axs[0].set_ylim([-10, 510])

    # Plot stddev range around average reward
    axs[0].fill_between(list(range(average_rewards.shape[0])), average_rewards - std_rewards, average_rewards + std_rewards)

    # Decorate plot
    axs[0].set_title("Rewards")
    axs[0].set_xlabel("Episode")
    axs[0].legend()

    ### Plot average losses ###
    axs[1].plot(average_losses, color="r", alpha=.7)

    # Plot stddev range around average loss
    axs[1].fill_between(list(range(average_losses.shape[0])), average_losses - std_losses, average_losses + std_losses)

    # Decorate plot
    axs[1].set_title("Losses")
    axs[1].set_xlabel("Episode")
    axs[1].legend()
    
    plt.show()

if __name__ == "__main__":
    plot_agent(sys.argv[1])