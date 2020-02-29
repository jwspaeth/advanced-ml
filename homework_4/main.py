#!/usr/bin/env python3

import pickle
import os
import sys
import json
import subprocess
import itertools

import matplotlib.pyplot as plt
import numpy as np

from config.config_handler import config_handler
from exceptions import MissingConfigArgException

def main():
    """Spits out training jobs for each configuration"""

    # Get cfg name
    cfg_name = get_cfg_name()

    # Create configuration handler
    cfg_handler = config_handler(cfg_name)

    if cfg_handler.get_mode() == "train":
        # Create index file
        create_index_log(cfg_handler)

    # Start a job for each hyperparameter
    print("Number of jobs to start: {}".format(cfg_handler.get_num_experiments()))
    for i in range(cfg_handler.get_num_experiments()):
        start_job(cfg_name, i, cfg_handler.get_mode())

def get_cfg_name():

    for arg in sys.argv:
        if "-cfg_name=" in arg:
            return arg.replace("-cfg_name=", "")

    raise MissingConfigArgException()

def get_exp_num():

    for arg in sys.argv:
        if "-exp_num=" in arg:
            return int(arg.replace("-exp_num=", ""))

def create_index_log(cfg_handler):
    """Write index to file that describes experiment hyperparameters"""

    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    batch_name = cfg_handler.get_experiment(0).save.experiment_batch_name
    fbase = "{}{}/".format(fbase, batch_name)
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    with open('{}index.txt'.format(fbase), 'w') as f:
        f.write("Number of experiments: {}\n".format(cfg_handler.get_num_experiments()))
        json.dump(cfg_handler.get_options(), f)
        f.write("\n\n")

        for i in range(cfg_handler.get_num_experiments()):
            individual_option = cfg_handler.get_option(i)
            f.write("\tExperiment {}: ".format(i))
            json.dump(individual_option, f)
            f.write("\n")

def start_job(config_name, experiment_num, mode):
    """
    Starts a job for the fed arguments. This takes the form of a subprocess,
    whether on a normal computer or supercomputer
    """

    print("Starting job: {}".format(experiment_num))

    # Decide which script to run
    if "-s" in sys.argv:
        script_to_run = ["sbatch", "supercomputer_job.sh", "-s"]
    else:
        script_to_run = ["./standard_job.sh"]

    # Build script with hyperparameters
    full_command = [
        *script_to_run,
        "-job",
        "-cfg_name={}".format(config_name),
        "-exp_num={}".format(experiment_num)
    ]

    if mode == "train":
        full_command.append("-train")
    elif mode == "eval":
        full_command.append("-eval")
    else:
        raise Exception("Error: configuration mode must be either train or eval")


    # Run chosen script with correct arguments
    process = subprocess.Popen(full_command)

    # Wait if not parallel
    if "-p" not in sys.argv:
        process.wait()

def evaluate(model=None):
    """
    Evaluation only works on existing models and cached results. The model can be fed as an argument,
    or reloaded through the configuration file. If reloaded, it will save results into
    the original experiment's directory. Can evaluate one experiment at a time.
    """

    # Get configuration values
    cfg_name = get_cfg_name()
    experiment_num = get_exp_num()
    cfg_handler = config_handler(cfg_name)
    exp_cfg = cfg_handler.get_experiment(experiment_num)

    # Load dataset
    dataset = cfg_handler.get_dataset(exp_cfg)

    # Load model
    if model is None:
        if exp_cfg.model.reload_path == "":
            raise Exception("Error: evaluation only works on existing model. Must model as an argument" +
                " or specify reload path to experiment")
        else:
            model = cfg_handler.get_model(exp_cfg=exp_cfg)

    # Load evaluation functions
    evaluation_functions = cfg_handler.get_evaluation_functions(exp_cfg)

    # Use evaluation functions
    for evaluation_function in evaluation_functions:
        evaluation_function(dataset, model, exp_cfg)

def train():

    # Get configuration values
    cfg_name = get_cfg_name()
    experiment_num = get_exp_num()
    cfg_handler = config_handler(cfg_name)
    exp_cfg = cfg_handler.get_experiment(experiment_num)

    # Define fbase and create tree
    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase += "{}/".format(exp_cfg.save.experiment_batch_name)
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase += "experiment_{}/".format(experiment_num)
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    # Create print mechanisms
    reset_log_files(cfg_handler, fbase)
    redirect_stdout(cfg_handler, fbase)
    redirect_stderr(cfg_handler, fbase)

    # Cache configuration for future reload
    save_cfg(exp_cfg, fbase)

    # Print info
    print("Config name: {}".format(cfg_name))
    print("Experiment num: {}".format(experiment_num))
    print()

    # Load data
    dataset = cfg_handler.get_dataset(exp_cfg)
    data_dict = dataset.load_data()

    print("Train ins shape: {}".format(data_dict["train"]["ins"].shape))
    print("Train outs shape: {}".format(data_dict["train"]["outs"].shape))

    # Build model
    model = cfg_handler.get_model(
        input_size=dataset.get_input_size(),
        exp_cfg=exp_cfg)

    # Compile model
    model.compile(
        optimizer=exp_cfg.train.optimizer,
        loss=exp_cfg.train.loss,
        metrics=exp_cfg.train.metrics)
    model.summary()

    # Callbacks
    callbacks = cfg_handler.get_callbacks(fbase, exp_cfg)

    # Future work
    # Check if train key is a dict. If so, do non-generator training. Else, do generator training.
    # Check if val key exists. If not, don't use.
    #   Otherwise, check if val is a dict or generator and proceed accordingly
    # In total, 6 options.

    # Train model. Requirements for model.fit are liable to expand in the future, and would
    #   require further development. e.g. using sample or class weights, validation steps, or validation frequency
    if "val" in data_dict.keys():
        history = model.fit(
                x=data_dict["train"]["ins"],
                y=data_dict["train"]["outs"],
                validation_data = (data_dict["val"]["ins"], data_dict["val"]["outs"]),
                epochs=exp_cfg.train.epochs,
                batch_size=exp_cfg.train.batch_size,
                callbacks=callbacks,
                verbose=exp_cfg.train.verbose
                )
    else:
        history = model.fit(
                x=data_dict["train"]["ins"],
                y=data_dict["train"]["outs"],
                epochs=exp_cfg.train.epochs,
                batch_size=exp_cfg.train.batch_size,
                callbacks=callbacks,
                verbose=exp_cfg.train.verbose
                )

    # Log results
    log_results(data_dict, model, exp_cfg, fbase)

def log_results(data, model, exp_cfg, fbase):
    """Log results to file"""

    print("Logging results")

    # Generate results
    results = {}
    results_brief = {}
    results["history"] = model.history.history
    if "train" in data.keys():
        results['predict_train'] = model.predict(data["train"]["ins"])
        results['eval_train'] = model.evaluate(data["train"]["ins"], data["train"]["outs"])
        results_brief["eval_train"] = str(results["eval_train"])

    if "val" in data.keys():
        results['predict_val'] = model.predict(data["val"]["ins"])
        results['eval_val'] = model.evaluate(data["val"]["ins"], data["val"]["outs"])
        results_brief["eval_val"] = str(results["eval_val"])

    if "test" in data.keys():
        results['predict_test'] = model.predict(data["test"]["ins"])
        results['eval_test'] = model.evaluate(data["test"]["ins"], data["test"]["outs"])
        results_brief["eval_test"] = str(results["eval_test"])

    # Create model directory
    if not os.path.exists("{}/model_and_cfg/".format(fbase)):
        os.mkdir("{}/model_and_cfg/".format(fbase))

    # Save model
    model.save("{}/model_and_cfg/".format(fbase), save_format="tf")

    # Save brief results for human readability
    with open("{}results_brief.txt".format(fbase), "w") as f:
        f.write(json.dumps(results_brief))

    # Save full results binary
    with open("{}results_dict.pkl".format(fbase), "wb") as f:
        pickle.dump(results, f)

def save_cfg(exp_cfg, fbase):
    # File path
    filename = fbase + "model_and_cfg/"
    if not os.path.exists(filename):
        os.mkdir(filename)

    filename += "exp_cfg"
    with open(filename, "wt") as f:
        f.write(exp_cfg.dump())

def reset_log_files(cfg_handler, fbase):
    # File path
    out_filename = fbase + "out_log.txt"
    with open(out_filename, "w") as f:
        pass

    err_filename = fbase + "err_log.txt"
    with open(err_filename, "w") as f:
        pass

def redirect_stdout(cfg_handler, fbase):

    filename = fbase + "out_log.txt"

    sys.stdout = open(filename, mode="a")

    print("--------Stdout Start--------")

def redirect_stderr(cfg_handler, fbase):

    filename = fbase + "err_log.txt"

    sys.stderr = open(filename, mode="a")

    print("--------Stderr Start--------", file=sys.stderr)

if __name__ == "__main__":

    # If this is a subprocess, run the training program
    if "-job" in sys.argv:
        if "-train" in sys.argv:
            train()
        elif "-eval" in sys.argv:
            evaluate()
    else:
        main()





