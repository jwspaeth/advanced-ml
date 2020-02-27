#!/usr/bin/env python3

import sys
import json
import pickle
import os
import glob

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

from datasets import Core50Dataset

def main():

    batch_name = "log_test"
    fbase = "results/{}/".format(batch_name)

    # Load experiment batch
    results = load_batch_results(fbase)

    # Create list of validation accuracy curves
    acc_curves = [result["history"]["val_acc"] for result in results]

    # Plot learning curves
    plot_learning_curves(fbase, acc_curves, "Validation")

    # Create list of ROC curves
    all_predictions = [result["predict_val"] for result in results]
    outs = Core50Dataset().load_data()["val"]["outs"]
    roc_curves = [generate_roc_curve(outs, predictions) for predictions in all_predictions]

    # Plot ROC curves
    plot_roc_curves(fbase, roc_curves, "Validation")

def load_batch_results(fbase):
    file_pattern = fbase + "/experiment*/"
    filepaths = glob.glob(file_pattern)

    results = []
    for filepath in filepaths:
        results.append( load_result_from_experiment(filepath) )

    return results

def load_result_from_experiment(fbase):
    """Load result of given experiment"""
    filename = fbase + "results_dict.pkl"
    with open(filename, "rb") as fp:
        return pickle.load(fp)

def generate_roc_curve(outs, predictions):
        '''
        Produce a ROC plot given a model, a set of inputs and the true outputs
        Assume that model produces N-class output; we will only look at the class 0 scores
        '''
        # Compute false positive rate & true positive rate + AUC
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(outs[:,0], predictions[:,0])
        auc = sklearn.metrics.auc(fpr, tpr)

        curve = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc
        }

        return curve

def plot_learning_curves(fbase, curves, set_name):
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for curve in curves:
        ax.plot(curve)
    plt.title("{} Learning Curve".format(set_name))
    plt.ylabel("Accuracy")
    plt.xlabel("Epochs")

    fig.savefig(fbase + "learning_curves.png", dpi=fig.dpi)

def plot_roc_curves(fbase, curves, set_name):
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    # Generate the plot
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for curve in curves:
        ax.plot(curve["fpr"], curve["tpr"], 'r', label='AUC = {:.3f}'.format(curve["auc"]))
    plt.title("{} ROC".format(set_name))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate') 

    fig.savefig(fbase + "roc_curves.png", dpi=fig.dpi)

if __name__ == "__main__":
    main()

