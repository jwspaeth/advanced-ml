#!/usr/bin/env python3

import pickle
import os
import sys

import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import numpy as np

from config.configuration_handler import configuration_handler

def dispatch_jobs():
    """ Dispatch jobs requested. n_jobs is computed in configuration file. """
    
    config_name = get_config_name()

    config_handler = configuration_handler(config_name)
    n_jobs = config_handler.get_n_jobs()

    for i in range(n_jobs):
        dispatch_job(i, config_name)

def dispatch_job(job_number, config_name):
    """
    Dispatch job with specific job number and configuration name. Job number determines which experiment is run
    from the combination of hyperparameters. Configuration name determines which model is chosen.
    """
    script_to_run = get_batch_script_command()
    process = subprocess.Popen([*script_to_run, "{}".format(job_number), config_name])

    if get_parallel():
        process.wait()

def get_config_name():
    return arg.replace("-cfg=", "")

def get_parallel():

    if "-p" in sys.argv:
        return True
    else:
        return False

def get_batch_script_command():

    if "-s" in sys.argv:
        return ["sbatch", "supercomputer_job.sh"]
    else:
        return ["./standard_job.sh"]

#########################

def dnn(hidden_sizes, hidden_act="sigmoid", output_act="tanh"):
    """Construct a simple deep neural network"""

    inputs = Input(shape=(8,))

    hidden_stack_out = hidden_stack(hidden_sizes, hidden_act)(inputs)

    outputs = Dense(1, activation=output_act)(hidden_stack_out)

    return Model(inputs=inputs, outputs=outputs)

def hidden_stack(hidden_sizes, hidden_act="sigmoid"):
    """Represents a stack of neural layers"""

    layers = []
    for size in hidden_sizes:
        layers.append(Dense(size, activation=hidden_act))

    def hidden_stack_layer(inputs):
        """Layer hook for stack"""

        for i in range(len(layers)):
            if i == 0:
                carry_out = layers[i](inputs)
            else:
                carry_out = layers[i](carry_out)

        return carry_out

    return hidden_stack_layer

def create_histogram(prediction_errors):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(prediction_errors, 50)
    plt.ylabel("Count")
    plt.xlabel("Error")

    fig.savefig("error_histogram.png", dpi=fig.dpi)

def plot_learning_curve(experiment_num, history):

    save_path = "learning_curves/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(history.history["loss"])
    plt.ylabel("MSE")
    plt.xlabel("Epochs")

    fig.savefig(save_path + f"experiment_{experiment_num}.png", dpi=fig.dpi)


if __name__ == "__main__":
    dispatch_jobs()





