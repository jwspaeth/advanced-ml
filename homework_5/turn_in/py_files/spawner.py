#!/usr/bin/env python3

import sys
import os
import subprocess

from solver import main

def start_job(agent_folder, trial_name):
    """
    Starts a job for the fed arguments. This takes the form of a subprocess,
    whether on a normal computer or supercomputer
    """

    print("Starting job: folder -> {}, trial -> {}".format(agent_folder, trial_name))

    # Decide which script to run
    if "-s" in sys.argv:
        script_to_run = ["sbatch", "supercomputer_job.sh"]
    else:
        script_to_run = ["./standard_job.sh"]

    # Build script with hyperparameters
    full_command = [
        *script_to_run,
        "{}".format(agent_folder),
        "{}".format(trial_name)
    ]

    # Run chosen script with correct arguments
    process = subprocess.Popen(full_command)

    # Wait if not parallel
    if "-p" not in sys.argv:
        process.wait()

def loop_agents_and_average(agent_folder="exp_1", n_trials=5):

    if os.path.exists(agent_folder):
        raise Exception("Folder {} already exists.".format(agent_folder))

    for trial in range(n_trials):
        start_job(agent_folder=agent_folder, trial_name="trial_{}".format(trial))

if __name__ == "__main__":

    n_trials = 5
    for arg in sys.argv:
        if "-agent_folder=" in arg:
            agent_folder = arg.replace("-agent_folder=", "")

        if "-n_trials=" in arg:
            n_trials = int(arg.replace("-n_trials=", ""))
            
    loop_agents_and_average(agent_folder=agent_folder, n_trials=n_trials)