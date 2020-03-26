#!/usr/bin/env python3

import statistics

import gym
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
import tensorflow.keras as keras

from models import cnn
from agents import DQN, TargetDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy

def main():

    # Create environment
    env = gym.make('CarRacing-v0')
    print("State space: {}".format(env.observation_space))
    print("State space shape: {}".format(env.observation_space.shape))
    print("Action space: {}".format(env.action_space)) # Steer (-1, 1), Gas (0, 1), Brake (0, 1)
    print("Action space shape: {}".format(env.action_space.shape))

    # Create agent configuration
    agent_class = DQN
    state_size = env.observation_space.shape
    action_size = env.action_space.shape
    policy = epsilon_greedy_policy
    loss_fn = keras.losses.mean_squared_error
    epsilon = epsilon_episode_decay(1, .01, 200)
    gamma = .95
    buffer_size = 10000
    model_fn = cnn
    model_param_dict = {
            "input_size": state_size,
            "filters": [10, 15],
            "kernels": [3, 3],
            "strides": [2, 2],
            "max_pool_sizes": [2, 2],
            "cnn_l2": 0,
            "dnn_hidden_sizes": [20],
            "dnn_l2": 0,
            "n_options": [3, 2, 2]
            }
    learning_rate = .001
    learning_delay = 0
    verbose = True
    target_update_freq = 25

    # Create silent episode configuration
    silent_episodes = CN()
    silent_episodes_n_episodes = 1
    silent_episodes_n_steps = None
    silent_episodes_render_flag = False
    silent_episodes_batch_size = 2000
    silent_episodes_verbose = True

    # Create visible episodes configuration
    visible_episodes = CN()
    visible_episodes_n_episodes = 5
    visible_episodes_n_steps = None
    visible_episodes_render_flag = True
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
        verbose=visible_episodes_verbose
        )

    # Plot results
    print("--Plotting--")
    fig, axs = plt.subplots(2, 2)
    
    # Take mean over last 100 episodes
    if len(agent.reward_log) >= 100:
        cut = 100
    else:
        cut = len(agent.reward_log)
    axs[0, 0].plot(agent.reward_log, label="Agent {} -- Avg: {:.2f}".format(agent.type,
        statistics.mean(agent.reward_log[len(agent.reward_log)-cut:])))
    #axs[0, 0].set_ylim([-550, 50])
    axs[0, 0].legend()
    
    axs[0, 1].plot(agent.deque_log, label="Deque Size")
    axs[0, 1].legend()

    pad = [0] * agent.learning_delay
    axs[1, 0].plot(pad + agent.loss_log, label="Loss")
    axs[1, 0].legend()

    axs[1, 1].plot(agent.epsilon_log, label="Epsilon")
    axs[1, 1].legend()

    plt.show()

if __name__ == "__main__":
    main()
