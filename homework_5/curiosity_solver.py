#!/usr/bin/env python3

import statistics

import gym
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN
import tensorflow.keras as keras

from agents import DQN, TargetDQN, CuriosityDQN
from policies import epsilon_episode_decay, random_policy, epsilon_greedy_policy_generator, acrobot_epsilon_decay


def main():

    # Create environment
    env = gym.make('Acrobot-v1')
    print("State space: {}".format(env.observation_space))
    print("Action space: {}".format(env.action_space))

    # Create agent configuration
    agent_class = CuriosityDQN
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    policy = epsilon_greedy_policy_generator(-1, 2)
    loss_fn = keras.losses.mean_squared_error
    epsilon = epsilon_episode_decay(1, .01, 200)
    #epsilon = 1
    gamma = .99
    buffer_size = 10000
    n_units = [16, 8]
    learning_rate = .01
    learning_delay = 0
    learning_freq = 1
    verbose = True
    curiosity_n_units = [4, 4]
    curiosity_weight = 2

    # Create silent episode configuration
    silent_episodes = CN()
    silent_episodes.n_episodes = 100
    silent_episodes.n_steps = 500
    silent_episodes.render_flag = False
    silent_episodes.batch_size = 2000
    silent_episodes.verbose = True

    # Create visible episodes configuration
    visible_episodes = CN()
    visible_episodes.n_episodes = 1
    visible_episodes.n_steps = 500
    visible_episodes.render_flag = True
    visible_episodes.batch_size = 2000
    visible_episodes.verbose = True

    # Build agent
    agent = agent_class(
        state_size=state_size,
        action_size=action_size,
        policy=policy,
        loss_fn=loss_fn,
        epsilon=epsilon,
        gamma=gamma,
        buffer_size=buffer_size,
        n_units=n_units,
        learning_rate=learning_rate,
        learning_delay=learning_delay,
        learning_freq=learning_freq,
        verbose=verbose,
        curiosity_n_units=curiosity_n_units,
        curiosity_weight=curiosity_weight
        )

    print("--Training--")
    print("\tAgent type: {}".format(agent.type))

    # Run silent episodes
    agent.execute_episodes(
        env=env,
        n_episodes=silent_episodes.n_episodes,
        n_steps=silent_episodes.n_steps,
        render_flag=silent_episodes.render_flag,
        batch_size=silent_episodes.batch_size,
        verbose=silent_episodes.verbose
        )

    # Run visible episodes
    agent.execute_episodes(
        env=env,
        n_episodes=visible_episodes.n_episodes,
        n_steps=visible_episodes.n_steps,
        render_flag=visible_episodes.render_flag,
        batch_size=visible_episodes.batch_size,
        verbose=visible_episodes.verbose,
        train=False
        )

    # Plot results
    print("--Plotting--")
    fig, axs = plt.subplots(4, 2)
    
    # Take mean over last 100 episodes
    if len(agent.reward_log) >= 100:
        cut = 100
    else:
        cut = len(agent.reward_log)
    axs[0, 0].plot(agent.reward_log, label="Agent {} -- Avg: {:.2f}".format(agent.type,
        statistics.mean(agent.reward_log[len(agent.reward_log)-cut:])))
    axs[0, 0].set_ylim([-550, 50])
    axs[0, 0].legend()
    
    axs[0, 1].plot(agent.deque_log, label="Deque Size")
    axs[0, 1].legend()

    pad = [0] * agent.learning_delay
    axs[1, 0].plot(pad + agent.loss_log, label="Loss")
    axs[1, 0].legend()

    axs[1, 1].plot(agent.curiosity_log, label="Curiosity")
    axs[1, 1].legend()

    axs[2, 0].plot(agent.curiosity_reward_log, label="Curiosity Reward")
    axs[2, 0].legend()

    axs[2, 1].plot(agent.raw_score_log, label="Raw Score")
    axs[2, 1].legend()

    axs[3, 0].plot(agent.q_val_log, label="Q Value")
    axs[3, 0].legend()

    plt.show()

if __name__ == "__main__":
    main()
