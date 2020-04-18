#!/usr/bin/env python3

import statistics
import sys
import os
import pickle

import gym
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
import tensorflow.keras as keras

from models import cnn, dnn, dueling_dnn
from agents import DQN, TargetDQN, DoubleDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy_car_generator
from policies import epsilon_greedy_policy_generator, epsilon_greedy_policy_lunar_lander

def save_results_and_models(agent, agent_folder, trial_name):
    fbase = "results/"
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase = "{}/".format(fbase + agent_folder)
    if not os.path.exists(fbase):
        os.mkdir(fbase)
    fbase = "{}/".format(fbase + trial_name)
    if not os.path.exists(fbase):
        os.mkdir(fbase)

    results = {}
    results["rewards"] = agent.reward_log
    results["losses"] = agent.loss_log
    print("Reward log length: {}".format(len(results["rewards"])))
    print("Loss log length: {}".format(len(results["losses"])))

    # Save full results binary
    with open("{}results_dict.pkl".format(fbase), "wb") as f:
        pickle.dump(results, f)

    if agent.type == "DQN":
        agent.model.save("{}model.h5".format(fbase))
    elif agent.type == "TargetDQN" or agent.type == "DoubleDQN":
        agent.model.save("{}model.h5".format(fbase))
        agent.target_model.save("{}target_model.h5".format(fbase))

def main():

    agent_folder = sys.argv[1]
    trial_name = sys.argv[2]

    keras.backend.clear_session()

    # Create environment
    env = gym.make('CartPole-v1')
    #env = gym.wrappers.Monitor("CartPole-v1", "results/V5/trial_15/")

    print("State space: {}".format(env.observation_space))
    print("State space shape: {}".format(env.observation_space.shape))
    print("Action space: {}".format(env.action_space))
    print("Action space shape: {}".format(env.action_space.shape))

    # Create agent configuration
    agent_class = DQN
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    policy = epsilon_greedy_policy_generator(0, 2)
    #policy = epsilon_greedy_policy_lunar_lander
    loss_fn = keras.losses.mean_squared_error
    epsilon = epsilon_episode_decay(1, .01, 200)
    gamma = .99
    buffer_size = 100000
    model_fn = dnn
    model_param_dict = {
            "input_size": state_size,
            "hidden_sizes": [15, 15],
            "hidden_act": "relu",
            "n_options": [2]
            }
    learning_rate = .001
    learning_delay = 0
    verbose = True
    reload_path = "results/V5/trial_29/model.h5"
    target_update_freq = 100

    # Create silent episode configuration
    silent_episodes = CN()
    silent_episodes_n_episodes = 0
    silent_episodes_n_steps = None
    silent_episodes_render_flag = False
    silent_episodes_batch_size = 32
    silent_episodes_verbose = True

    # Create visible episodes configuration
    visible_episodes = CN()
    visible_episodes_n_episodes = 2
    visible_episodes_n_steps = None
    visible_episodes_render_flag = True
    visible_episodes_batch_size = 32
    visible_episodes_verbose = True

    # Build agent
    agent = agent_class(
        state_size=state_size,
        action_size=action_size,
        policy=policy,
        loss_fn=loss_fn,
        epsilon=epsilon,
        gamma=gamma,
        buffer_size=buffer_size,
        model_fn=model_fn,
        model_param_dict=model_param_dict,
        learning_rate=learning_rate,
        learning_delay=learning_delay,
        verbose=verbose,
        reload_path=reload_path,
        target_update_freq=target_update_freq
        )

    print("--Training--")
    print("\tAgent type: {}".format(agent.type))

    # Run silent episodes
    agent.execute_episodes(
        env=env,
        n_episodes=silent_episodes_n_episodes,
        n_steps=silent_episodes_n_steps,
        render_flag=silent_episodes_render_flag,
        batch_size=silent_episodes_batch_size,
        verbose=silent_episodes_verbose
        )

    # Run visible episodes
    agent.execute_episodes(
        env=env,
        n_episodes=visible_episodes_n_episodes,
        n_steps=visible_episodes_n_steps,
        render_flag=visible_episodes_render_flag,
        batch_size=visible_episodes_batch_size,
        verbose=visible_episodes_verbose,
        train=False
        )

    save_results_and_models(agent, agent_folder, trial_name)

if __name__ == "__main__":
    main()
