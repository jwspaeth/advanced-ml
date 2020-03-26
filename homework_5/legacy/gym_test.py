#!/usr/bin/env python3

import time

import gym

def basic_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

episode_rewards = 0
env = gym.make("CartPole-v1")
obs = env.reset()
for step in range(1000):
	env.render()
	action = basic_policy(obs)
	obs, reward, done, info = env.step(action)
	episode_rewards += reward

env.close()