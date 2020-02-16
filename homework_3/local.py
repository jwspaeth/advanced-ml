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

    # Load dropout experiments and results
    dropout_options, dropout_experiments = load_index_log(batch_num=0)
    dropout_results = [load_result_from_experiment(experiment) for experiment in dropout_experiments]

    # Load l2 experiments and results
    l2_options, l2_experiments = load_index_log(batch_num=1)
    l2_results = [load_result_from_experiment(experiment) for experiment in l2_experiments]

    # Compute fvaf curves for each key value
    '''dropout_avg_fvafs = compute_avg_fvaf_curves(dropout_results, dropout_options["dropout"], dropout_options["n_train_folds"])
    l2_avg_fvafs = compute_avg_fvaf_curves(l2_results, l2_options["dropout"], dropout_options["n_train_folds"])

    # Plot fvaf curves for each key value
    plot_fvaf_curves(dropout_avg_fvafs["val"], dropout_options["dropout"], dropout_option["n_train_folds"])
    plot_fvaf_curves(l2_avg_fvafs["val"], l2_options["l2"], l2_options["n_train_folds"])

    # Get argmax for the best hyperparameter values
    dropout_argmax_fvafs = np.amax(dropout_avg_fvafs["val"], axis=0)
    l2_argmax_fvafs = np.amax(l2_avg_fvafs["val"], axis=0)

    # Plot the test set fvaf for the argmaxes
    plot_fvaf(dropout_avg_fvafs["test"][dropout_argmax_fvafs])
    plot_fvaf(l2_avg_fvafs["test"][l2_argmax_fvafs])'''

def load_result_from_experiment(experiment):
    file_str = "results/batch_{}/experiment_{}/results_dict.pkl".format(experiment["batch_num"], experiment["experiment_num"])
    with open(file_str) as fp:
        return pickle.load(fp)

def load_index_log(batch_num):

    experiments = []
    with open("results/batch_{}/index.txt".format(batch_num), "r") as f:
        contents = f.read().split("\n")

        options = json.loads(contents[1])
        for experiment_str in contents[2:]:
            experiments.append(json.loads(experiment_str))

        return options, experiments

def plot_fvaf(avg_fvafs, n_train_folds_list, set_name):
    """Plot fvaf based on the set name"""

    # Create results directory
    save_path = "results/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Create plots directory
    save_path += "fvaf_plots/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Create and configure plot
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

    # Save
    fig.savefig("{}{}_fvaf_plot.png".format(save_path, set_name), dpi=fig.dpi)

def compute_avg_fvaf_curves(results, key, key_list, n_train_folds):

    curve_list = []
    for key_val in key_list:
        key_results = [results if results[key]]
        curve_list.append( compute_avg_fvafs(results) )


def compute_avg_fvafs(results, n_train_folds):
    
    n_train_folds_list = options["n_train_folds"]

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

