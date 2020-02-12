#!/usr/bin/env python3

import sys
import json
import pickle
import os
import subprocess

import matplotlib.pyplot as plt

def main():

    # Use -s argument to scp results from supercomputer before continuing
    if "-s" in sys.argv:
        script_to_run = [
            "scp",
            "-r", 
            "jwspaeth@schooner.oscer.ou.edu:/home/jwspaeth/workspaces/advanced-ml/homework_2/results",
            "./"]
        process = subprocess.Popen([*script_to_run])
        process.wait()

    # Read index file
    index = load_index_log()
    rotation_list = index["rotation_list"]
    n_train_folds_list = index["n_train_folds_list"]

    # Load all results
    results = []
    for rotation in rotation_list:
        for n_train_folds in n_train_folds_list:
            with open("results/r{:02d}_t{:02d}/results.pkl".format(rotation, n_train_folds), "rb") as fp:
                results.append(pickle.load(fp))

    # Compute average fvafs
    avg_fvafs = compute_avg_fvafs(results, n_train_folds_list)

    # Plot and save all the fvafs
    plot_fvaf(avg_fvafs, n_train_folds_list, "train")
    plot_fvaf(avg_fvafs, n_train_folds_list, "val")
    plot_fvaf(avg_fvafs, n_train_folds_list, "test")

def load_index_log():

    with open("results/index.json") as f:
        return json.load(f)

def plot_fvaf(avg_fvafs, n_train_folds_list, set_name):
    
    save_path = "results/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path += "fvaf_plots/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(n_train_folds_list, avg_fvafs[set_name])
    plt.ylabel("Average FVAF")
    plt.xlabel("Number of Training Folds")

    if set_name == "train":
        plt.title("Training Set")
    elif set_name == "val":
        plt.title("Validation Set")
    elif set_name == "test":
        plt.title("Test Set")

    fig.savefig("{}{}_fvaf_plot.png".format(save_path, set_name), dpi=fig.dpi)

def compute_avg_fvafs(results, n_train_folds_list):
    
    # Sum all the fvafs and count how many values there are
    # Each index represents a n_train_folds hyperparameter
    avg_fvafs = {
        "train": [0]*len(n_train_folds_list),
        "val": [0]*len(n_train_folds_list),
        "test": [0]*len(n_train_folds_list)
    }

    # Loop through each split
    for key in avg_fvafs.keys():

        # Start summing and count the fvaf values
        sum_fvafs = [0]*len(n_train_folds_list)
        count_fvafs = [0]*len(n_train_folds_list)
        for i in range(len(n_train_folds_list)):

            for result in results:
                if result["n_train_folds"] == n_train_folds_list[i]:
                    sum_fvafs[i] += result["eval_{}".format(key)][1]
                    count_fvafs[i] += 1

        # Create average fvafs based on the sum and counts
        for i in range(len(n_train_folds_list)):
            if count_fvafs[i] != 0:
                avg_fvafs[key][i] = sum_fvafs[i] / count_fvafs[i]

    return avg_fvafs

if __name__ == "__main__":
    main()

