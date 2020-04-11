#!/usr/bin/env python3

import statistics
import sys
import os
import pickle

import gym
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
import tensorflow.keras as keras

from models import cnn, dnn
from agents import DQN, TargetDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy_car_generator
from policies import epsilon_greedy_policy_generator

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
    elif agent.type == "TargetDQN":
        agent.model.save("{}model.h5".format(fbase))
        agent.target_model.save("{}target_model.h5".format(fbase))

def main():

    agent_folder = sys.argv[1]
    trial_name = sys.argv[2]

    keras.backend.clear_session()

    # Create environment
    env = gym.make('CartPole-v1')
    print("State space: {}".format(env.observation_space))
    print("State space shape: {}".format(env.observation_space.shape))
    print("Action space: {}".format(env.action_space)) # Steer (-1, 1), Gas (0, 1), Brake (0, 1)
    print("Action space shape: {}".format(env.action_space.shape))

    # Create agent configuration
    agent_class = DQN
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    policy = epsilon_greedy_policy_generator(0, 2)
    loss_fn = keras.losses.mean_squared_error
    epsilon = epsilon_episode_decay(1, .01, 200)
    gamma = .95
    buffer_size = 10000
    model_fn = dnn
    model_param_dict = {
            "input_size": state_size,
            "hidden_sizes": [20],
            "n_options": [2]
            }
    learning_rate = .001
    learning_delay = 0
    verbose = True
    target_update_freq = 25

    # Create silent episode configuration
    silent_episodes = CN()
    silent_episodes_n_episodes = 500
    silent_episodes_n_steps = None
    silent_episodes_render_flag = False
    silent_episodes_batch_size = 2000
    silent_episodes_verbose = True

    # Create visible episodes configuration
    visible_episodes = CN()
    visible_episodes_n_episodes = 1
    visible_episodes_n_steps = None
    visible_episodes_render_flag = False
    visible_episodes_batch_size = 2000
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
