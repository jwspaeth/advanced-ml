#!/usr/bin/env python3

from collections import deque
import pickle
import os
import sys
import json
import subprocess

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

from models import dnn
from symbiotic_metrics import FractionOfVarianceAccountedFor

def main():

    rotation_list = list(range(20))
    n_train_folds_list = [1, 2, 3, 5, 10, 18]

    create_index_log(rotation_list, n_train_folds_list)

    for rotation in rotation_list:
        for n_train_folds in n_train_folds_list:
            start_training_job(rotation, n_train_folds)

def start_training_job(rotation, n_train_folds):

    if "-s" in sys.argv:
        script_to_run = ["sbatch", "supercomputer_job.sh", "-s"]
    else:
        script_to_run = ["./standard_job.sh"]

    process = subprocess.Popen(
        [*script_to_run, 
        "-job", # Indicate to the subprocess that it is a subprocess
        "-rotation={}".format(rotation),
        "-n_train_folds={}".format(n_train_folds)
        ])

    if "-p" not in sys.argv:
        process.wait()

def parse_args():

    for arg in sys.argv:
        if "-rotation=" in arg:
            rotation = int(arg.replace("-rotation=", ""))
        elif "-n_train_folds=" in arg:
            n_train_folds = int(arg.replace("-n_train_folds=", ""))

    return rotation, n_train_folds

def train(rotation=0, n_train_folds=18):
    
    # Rotate indices based on current rotation
    rotation_indices = get_rotation_indices(n_folds=20, rotation=rotation)

    # Get the training, validation, and test fold indices
    fold_inds = get_set_indices(rotation_indices=rotation_indices, n_train_folds=n_train_folds)

    ''' Load data
    Key MI, Length 20, Shape (1193, 960)
    Key theta, Length 20, Shape (1193, 2)
    Key dtheta, Length 20, Shape (1193, 2)
    Key ddtheta, Length 20, Shape (1193, 2)
    Key torque, Length 20, Shape (1193, 2)
    Key time, Length 20, Shape (1193, 1)
    '''
    if "-s" in sys.argv:
        data_path = "/home/fagg/ml_datasets/bmi/bmi_dataset.pkl"
    else:
        data_path = "bmi_dataset.pkl"
    with open(data_path, "rb") as fp:
        hw2_dataset = pickle.load(fp)

    # Splits the data into its respective train, validation, and test sets / ins and outs
    processed_data = process_dataset(hw2_dataset, fold_inds)

    # Build model
    model = dnn(
        input_size=(processed_data["train"]["ins"].shape[1],),
        hidden_sizes=[100, 50],
        output_size=processed_data["train"]["outs"].shape[1],
        hidden_act="elu",
        output_act="linear")

    # Compile model with fvaf metric
    fvaf = FractionOfVarianceAccountedFor(processed_data["test"]["outs"].shape[1])
    model.compile(optimizer="adam", loss="mse", metrics=[fvaf], verbose=2)
    model.summary()

    # Callbacks
    es_callback = EarlyStopping(
                            monitor="val_loss",
                            patience=5,
                            restore_best_weights=True,
                            min_delta=.0001)

    # Train model
    history = model.fit(
            x=processed_data["train"]["ins"],
            y=processed_data["train"]["outs"],
            validation_data = (processed_data["val"]["ins"], processed_data["val"]["outs"]),
            epochs=100,
            batch_size=32,
            callbacks=[es_callback]
            )

    # Log results
    log(model, processed_data, fold_inds, rotation, n_train_folds)

    # Plot the torque and save figure
    plot_torque(model, processed_data, rotation, n_train_folds)

def plot_torque(model, data, rotation, n_train_folds):

    save_path = "results/"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    save_path += "r{}_t{}/".format(rotation, n_train_folds)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    true_torque = data["test"]["outs"][:, 0]
    predicted_torque = model.predict(data["test"]["ins"])[:, 0]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data["test"]["time"], true_torque, label="True Torque")
    ax.plot(data["test"]["time"], predicted_torque, label="Predicted Torque")
    ax.legend()
    plt.ylabel("Torque")
    plt.xlabel("Time")

    fig.savefig(save_path + f"torque_plot.png", dpi=fig.dpi)

def create_index_log(rotation_list, n_train_folds_list):
    
    index = {
        "rotation_list": rotation_list,
        "n_train_folds_list": n_train_folds_list
    }

    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    with open('{}index.json'.format(fbase), 'w') as f:
        json.dump(index, f)

def log(model, data, fold_inds, rotation, n_train_folds):
    print("Logging results")

    # Generate results
    results = {}
    results['predict_train'] = model.predict(data["train"]["ins"])
    results['eval_train'] = model.evaluate(data["train"]["ins"], data["train"]["outs"])
    results['predict_val'] = model.predict(data["val"]["ins"])
    results['eval_val'] = model.evaluate(data["val"]["ins"], data["val"]["outs"])
    results['predict_test'] = model.predict(data["test"]["ins"])
    results['eval_test'] = model.evaluate(data["test"]["ins"], data["test"]["outs"])
    results['folds'] = fold_inds
    results['rotation'] = rotation
    results['n_train_folds'] = n_train_folds

    # Create results directory
    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase += "r{}_t{}/".format(rotation, n_train_folds)
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    # Save results
    with open("{}results.pkl".format(fbase, rotation, n_train_folds), "wb") as fp:
        pickle.dump(results, fp)
        fp.close()

    # Create model directory
    if not os.path.exists("{}/model/".format(fbase)):
        os.mkdir("{}/model/".format(fbase))

    # Save model
    model.save("{}/model/".format(fbase))

def process_dataset(dataset, fold_inds):

    processed_data = {}
    for key in fold_inds.keys():
        processed_data[key] = split_dataset(dataset, fold_inds[key])

    print("Train ins shape: {}".format(processed_data["train"]["ins"].shape))
    print("Train outs shape: {}".format(processed_data["train"]["outs"].shape))

    return processed_data

def split_dataset(dataset, inds):
    
    processed_data = {
        "ins": None,
        "outs": [],
        "time": None
    }
    
    for key in dataset.keys():
        # Get folds for this key
        folds = [dataset[key][ind] for ind in inds]

        # Join the folds
        joined = np.concatenate((folds), axis=0)

        # See if the key is for the ins or outs of the dataset
        if key == "MI":
            processed_data["ins"] = joined
        elif key == "time":
            processed_data["time"] = joined
        elif key == "torque":
            processed_data["outs"] = np.expand_dims(joined[:, 1], axis=1)

    return processed_data

def get_set_indices(rotation_indices, n_train_folds):

    inds = {}
    inds["train"] = [rotation_indices[i] for i in range(n_train_folds)]
    inds["val"] = [rotation_indices[len(rotation_indices)-2]]
    inds["test"] = [rotation_indices[len(rotation_indices)-1]]

    return inds

def get_rotation_indices(n_folds, rotation=0):

    fold_list = list(range(n_folds))
    fold_list = deque(fold_list)
    fold_list.rotate(rotation)
    fold_list = list(fold_list)

    return fold_list

if __name__ == "__main__":

    if "-job" in sys.argv:
        rotation, n_train_folds = parse_args()
        train(rotation=rotation, n_train_folds=n_train_folds)
    else:
        main()





