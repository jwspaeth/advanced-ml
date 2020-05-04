#!/usr/bin/env python3

import statistics
import sys
import os
import pickle
import json

import gym
from gym import wrappers
import matplotlib.pyplot as plt
import tensorflow.keras as keras

from models import cnn, dnn, dueling_dnn
from agents import DQN, TargetDQN, DoubleDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy_car_generator
from policies import epsilon_greedy_policy_generator, epsilon_greedy_policy_lunar_lander
from policies import epsilon_exponential_decay

def save_results_and_models(agent, agent_folder, trial_name):
    fbase = setup_results_directory(agent_folder, trial_name)

    results = {}
    results["rewards"] = agent.reward_log
    results["losses"] = agent.loss_log
    results["epsilons"] = agent.epsilon_log
    results["deques"] = agent.deque_log
    print("Reward log length: {}".format(len(results["rewards"])))
    print("Loss log length: {}".format(len(results["losses"])))

    # Save best results
    with open("{}best_results.txt".format(fbase), "w") as f:
        f.write("Best episode: {}\n".format(agent.best_episode))
        f.write("Best average reward: {}\n".format(agent.best_average_reward))

    # Save full results binary
    with open("{}results_dict.pkl".format(fbase), "wb") as f:
        pickle.dump(results, f)

    '''
    if agent.type == "DQN":
        agent.model.save("{}model".format(fbase))
    elif agent.type == "TargetDQN" or agent.type == "DoubleDQN":
        agent.model.save("{}model".format(fbase))
        agent.target_model.save("{}target_model".format(fbase))
    '''

    agent.model.save_weights("{}/model/weights".format(fbase))
    with open("{}/model/model_config.json".format(fbase), "w") as f:
        json.dump(agent.model_config, f)

def setup_results_directory(agent_folder, trial_name):

    # Create core directory
    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase = "{}/".format(fbase + agent_folder)
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase = "{}/".format(fbase + trial_name)
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    # Create model directory
    if not os.path.exists("{}model/".format(fbase)):
        os.mkdir("{}model/".format(fbase))

    return fbase

def main():

    # Get arguments and setup results directory
    agent_folder = sys.argv[1]
    trial_name = sys.argv[2]
    fbase = setup_results_directory(agent_folder, trial_name)

    # Clear any models that may have previously existed this session
    keras.backend.clear_session()

    # Create environment
    #env = gym.make("CartPole-v1")
    env = gym.make('LunarLander-v2')
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
        "policy": epsilon_greedy_policy_generator(0, 4),
        "loss_fn": keras.losses.mean_squared_error,
        "epsilon": epsilon_episode_decay(1, .01, 500),
        #"epsilon": 0,
        "gamma": .99,
        "buffer_size": 500000,
        "model_config": {
            "model_fn": "dnn",
            "input_size": env.observation_space.shape,
            "hidden_sizes": [512, 256, 4],
            "hidden_act": "relu",
            "n_options": [4]
        },
        "learning_rate": .001,
        "learning_delay": 50,
        "verbose": True,
        "reload_path": None,
        "target_update_freq": 75
    }
    print("Agent config: {}".format(agent_config))

    # Create silent episode configuration
    execution_config = {
        "env": env,
        "n_episodes": 60,
        "n_steps": None,
        "render_flag": False,
        "batch_size": 64,
        "verbose": True,
        "train": True
    }
    print("Execution config: {}".format(execution_config))

    # Build agent
    agent = agent_config["agent_class"](**agent_config)

    # Print training start signal
    print("--Training--")
    print("\tAgent type: {}".format(agent.type))
    print("\tPython type: {}".format(type(agent)))

    # Run silent episodes
    agent.execute_episodes(**execution_config)

    # Cache results
    save_results_and_models(agent, agent_folder, trial_name)

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
