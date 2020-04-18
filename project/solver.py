#!/usr/bin/env python3

import statistics
import sys
import os
import pickle

import gym
from gym import wrappers
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from models import cnn, dnn, dueling_dnn
from agents import DQN, TargetDQN, DoubleDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy_car_generator
from policies import epsilon_greedy_policy_generator, epsilon_greedy_policy_lunar_lander

def save_results_and_models(agent, agent_folder, trial_name):
    fbase = setup_results_directory(agent_folder, trial_name)

    results = {}
    results["rewards"] = agent.reward_log
    results["losses"] = agent.loss_log
    print("Reward log length: {}".format(len(results["rewards"])))
    print("Loss log length: {}".format(len(results["losses"])))

    # Save best results
    with open("{}best_results.txt".format(fbase), "w") as f:
        f.write("Best episode: {}\n".format(agent.best_episode))
        f.write("Best reward total: {}\n".format(agent.best_reward_total))

    # Save full results binary
    with open("{}results_dict.pkl".format(fbase), "wb") as f:
        pickle.dump(results, f)

    if agent.type == "DQN":
        agent.model.save("{}model.h5".format(fbase))
    elif agent.type == "TargetDQN" or agent.type == "DoubleDQN":
        agent.model.save("{}model.h5".format(fbase))
        agent.target_model.save("{}target_model.h5".format(fbase))

def setup_results_directory(agent_folder, trial_name):
    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase = "{}/".format(fbase + agent_folder)
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase = "{}/".format(fbase + trial_name)
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    return fbase

def main():

    # Get arguments and setup results directory
    agent_folder = sys.argv[1]
    trial_name = sys.argv[2]
    fbase = setup_results_directory(agent_folder, trial_name)

    # Clear any models that may have previously existed this session
    keras.backend.clear_session()

    # Create environment
    env = gym.make('CartPole-v1')
    if "-v" in sys.argv:
        env = wrappers.Monitor(env, fbase)

    # Print information about environment
    print("State space: {}".format(env.observation_space))
    print("State space shape: {}".format(env.observation_space.shape))
    print("Action space: {}".format(env.action_space))
    print("Action space shape: {}".format(env.action_space.shape))

    # Create agent configuration
    agent_config = {
        "agent_class": DQN,
        "state_size": env.observation_space.shape,
        "action_size": env.action_space.shape,
        "policy": epsilon_greedy_policy_generator(0, 2),
        "loss_fn": keras.losses.mean_squared_error,
        "epsilon": epsilon_episode_decay(1, .01, 500),
        "gamma": .95,
        "buffer_size": 2000,
        "model_fn": dnn,
        "model_param_dict": {
            "input_size": env.observation_space.shape,
            "hidden_sizes": [32, 32],
            "hidden_act": "elu",
            "n_options": [2]
        },
        "learning_rate": .001,
        "learning_delay": 0,
        "verbose": True,
        "reload_path": None,
        "target_update_freq": 25
    }

    # Create silent episode configuration
    execution_config = {
        "env": env,
        "n_episodes": 10,
        "n_steps": None,
        "render_flag": False,
        "batch_size": 32,
        "verbose": True,
        "train": True
    }

    # Build agent
    agent = agent_config["agent_class"](**agent_config)

    # Print training start signal
    print("--Training--")
    print("\tAgent type: {}".format(agent.type))

    # Run silent episodes
    agent.execute_episodes(**execution_config)

    # Cache results
    save_results_and_models(agent, agent_folder, trial_name)

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
