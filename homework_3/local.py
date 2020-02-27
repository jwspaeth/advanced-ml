#!/usr/bin/env python3

import sys
import json
import pickle
import os
import subprocess

import matplotlib.pyplot as plt
import numpy as np

def main():

    # Use -s argument to scp results from supercomputer before continuing
    if "-s" in sys.argv:
        script_to_run = [
            "scp",
            "-r", 
            "jwspaeth@schooner.oscer.ou.edu:/home/jwspaeth/workspaces/advanced-ml/homework_3/results",
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
    #   (training folds x hyperparameter value), so average across rotations
    dropout_avg_fvafs = compute_avg_fvaf_curves(dropout_results,
                                                dropout_options["n_train_folds"],
                                                dropout_options["dropout"],
                                                "dropout",
                                                dropout_options["rotation"])
    l2_avg_fvafs = compute_avg_fvaf_curves(l2_results,
                                            l2_options["n_train_folds"],
                                            l2_options["l2"],
                                            "l2",
                                            l2_options["rotation"])

    # Plot fvaf curves for each key value
    plot_fvaf_curves(dropout_options["n_train_folds"],
                    dropout_options["dropout"],
                    dropout_avg_fvafs["val"],
                    "Validation Dropout")
    plot_fvaf_curves(l2_options["n_train_folds"],
                    l2_options["l2"],
                    l2_avg_fvafs["val"],
                    "Validation L2")

    # Get argmax for the best hyperparameter values
    dropout_argmax_fvafs = np.argmax(dropout_avg_fvafs["val"], axis=1)
    l2_argmax_fvafs = np.argmax(l2_avg_fvafs["val"], axis=1)

    # Build arrays containing test values for best validation models
    test_dropout = []
    test_l2 = []
    for i in range(dropout_argmax_fvafs.shape[0]):
        test_dropout.append(dropout_avg_fvafs["test"][i, dropout_argmax_fvafs[i]])
        test_l2.append(l2_avg_fvafs["test"][i, l2_argmax_fvafs[i]])
    test_dropout = np.expand_dims(np.concatenate(test_dropout), axis=1)
    test_l2 = np.expand_dims(np.concatenate(test_l2), axis=1)
    test = np.concatenate((test_dropout, test_l2), axis=1)

    # Plot the test set fvaf for the argmaxes
    plot_fvaf_curves(dropout_options["n_train_folds"],
                    ["Dropout", "L2"],
                    test,
                    "Test Curves")

def load_result_from_experiment(experiment):
    """Load result of given experiment"""

    file_str = "results/batch_{}/experiment_{}/results_dict.pkl".format(experiment["batch_num"], experiment["experiment_num"])
    with open(file_str, "rb") as fp:
        return pickle.load(fp)

def load_index_log(batch_num):
    """Load the index log of the batch"""

    experiments = []
    with open("results/batch_{}/index.txt".format(batch_num), "r") as f:
        contents = f.read().split("\n")

        options = json.loads(contents[1])
        for i, experiment_str in enumerate(contents[2:len(contents)-1]):
            experiments.append(json.loads(experiment_str))

        return options, experiments

def plot_fvaf_curves(n_train_folds_list, key_list, curves, plot_name):
    """Plot fvaf based on the given curves"""

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
    for i in range(curves.shape[1]):
        ax.plot(n_train_folds_list, curves[:, i], label=str(key_list[i]))
    plt.legend()
    plt.ylabel("Average FVAF")
    plt.xlabel("Number of Training Folds")

    plt.title(plot_name)

    # Save
    fig.savefig("{}{}_fvaf_plot.png".format(save_path, plot_name), dpi=fig.dpi)

def get_matching_result(results, n_train_fold, key_val, key_name, rotation_val):
    """Find the result that matches the given values"""

    for i, result in enumerate(results):

        if (int(result["n_train_folds"]) == n_train_fold) and (float(result[key_name]) == key_val) and (int(result["rotation"]) == rotation_val):
            return result

def compute_avg_fvaf_curves(results, n_train_fold_list, key_list, key_name, rotation_list):
    """Gets the average fvaf across rotations for both the validation and test sets"""

    # Create 3d array of results
    fvaf_curves = {
        "val": np.zeros(shape=(len(n_train_fold_list), len(key_list), len(rotation_list)), dtype=object),
        "test": np.zeros(shape=(len(n_train_fold_list), len(key_list), len(rotation_list)), dtype=object)
    }
    for i, n_train_fold in enumerate(n_train_fold_list):
        for j, key_val in enumerate(key_list):
            for k, rotation_val in enumerate(rotation_list):

                # Get result with the matching parameters
                result = get_matching_result(results, n_train_fold, key_val, key_name, rotation_val)

                # Store the validation or test fvaf
                fvaf_curves["val"][i, j, k] = result["eval_val"][1]
                fvaf_curves["test"][i, j, k] = result["eval_test"][1]

    # Average across rotations
    fvaf_curves["val"] = np.average(fvaf_curves["val"], axis=2)
    fvaf_curves["test"] = np.average(fvaf_curves["test"], axis=2)

    return fvaf_curves

if __name__ == "__main__":
    main()

