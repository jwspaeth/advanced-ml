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
from agents import DQN, TargetDQN, DoubleDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy_car_generator
from policies import epsilon_greedy_policy_generator


""" Experiments for cartpole
• DQN
--Pick gamma--
(1) Arc [32]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
(2) Arc [32]; gamma .99; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
# FAIL
--Try higher learning rate
(3) Arc [32]; gamma .95; lr .01; batch size 2000; Decay 1->.01 in 300; episodes 1000
(4) Arc [32]; gamma .99; lr .01; batch size 2000; Decay 1->.01 in 300; episodes 1000
--Pick architecture--
(5) Arc [40]; gamma .99; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
(6) Arc [16]; gamma .99; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
(7) Arc [16, 8]; gamma .99; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
(8) Arc [16, 8, 4]; gamma .99; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
--Resample gamma--
(9) Arc [16, 8, 4]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
--Try higher epsilon--
(10) Arc [16, 8, 4]; gamma .99; lr .001; batch size 2000; Decay 1->.1 in 300; episodes 1000
--Random--
(11) Arc [16, 8, 4]; gamma .99; lr .01; batch size 2000; Decay 1->.01 in 300; episodes 1000
(12) Arc [16, 8, 4]; gamma .99; lr .0001; batch size 2000; Decay 1->.01 in 300; episodes 1000
--https://github.com/gsurma/cartpole--
(13) Arc [24, 24]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000
(14) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000
• TargetDQN
(15) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; update 25
• DQN
(16) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; act relu <--
• TargetDQN
(17) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; act relu; update 25
(18) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; act relu; update 1
(19) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; act relu; update 1
(20) Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; act relu; update 50
• DQN
(21) Arc [24, 24]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000; act relu
• TargetDQN
(22) Arc [24, 24]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000; act relu; update 25
(23) Arc [24, 24]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000; act relu; update 10
(24) Arc [24, 24]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000; act relu; update 50
(25) Arc [24, 24]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000; act relu; update 25
• DQN
(26) Arc [16, 8, 4]; gamma .95; lr .001; batch size 2000; Decay 1->.01 in 300; episodes 1000; act relu
"""

""" Examples
DQN: Arc [24, 24]; gamma .95; lr .001; batch size 20; Decay 1->.01 in 300; episodes 1000; act relu
TargetDQN: Repeat previous with update 25
DoubleDQN: Repeat previous
"""

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

    print("State space: {}".format(env.observation_space))
    print("State space shape: {}".format(env.observation_space.shape))
    print("Action space: {}".format(env.action_space))
    print("Action space shape: {}".format(env.action_space.shape))

    # Create agent configuration
    agent_class = DQN
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    policy = epsilon_greedy_policy_generator(0, 2)
    loss_fn = keras.losses.mean_squared_error
    epsilon = epsilon_episode_decay(1, .01, 300)
    gamma = .99
    buffer_size = 100000
    model_fn = dnn
    model_param_dict = {
            "input_size": state_size,
            "hidden_sizes": [15],
            "hidden_act": "relu",
            "n_options": [2]
            }
    learning_rate = .001
    learning_delay = 0
    verbose = True
    target_update_freq = 25

    # Create silent episode configuration
    silent_episodes = CN()
    silent_episodes_n_episodes = 1000
    silent_episodes_n_steps = None
    silent_episodes_render_flag = False
    silent_episodes_batch_size = 32
    silent_episodes_verbose = True

    # Create visible episodes configuration
    visible_episodes = CN()
    visible_episodes_n_episodes = 1
    visible_episodes_n_steps = None
    visible_episodes_render_flag = False
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
