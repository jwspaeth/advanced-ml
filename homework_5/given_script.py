#!/usr/bin/env python3

import gym
import matplotlib.pyplot as plt

from given_classes import myAgent

def main():

    FONTSIZE = 18
    FIGURE_SIZE = (10,4)
    FIGURE_SIZE2 = (10,10)

    # Configure parameters
    plt.rcParams.update({'font.size': FONTSIZE, 'figure.figsize': FIGURE_SIZE})

    # Default tick label size
    plt.rcParams['xtick.labelsize'] = FONTSIZE
    plt.rcParams['ytick.labelsize'] = FONTSIZE

    env = gym.make('CartPole-v1')

    print("State space: {}".format(env.osveration_space.shape))

    # Cart-pole is a discrete action environment (provided continous values are dummies)
    agent = myAgent(env.observation_space.shape[0], env.action_space.n, None, gamma=0.99, epsilon=0.1, lrate=.001)
    agent.build_model([32, 32])

    # Execute n trials silently
    agent.execute_ntrials(env, 100, 1000, render_flag=False, batch_size=2000)

    # Execute trials while rendering
    agent.execute_ntrials(env, 10, 1000, render_flag=True, batch_size=2000)

    # Show accumulated reward as a function of trial
    plt.plot(agent.reward_log)

if __name__ == "__main__":
    main()