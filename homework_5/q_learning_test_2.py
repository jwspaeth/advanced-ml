#!/usr/bin/env python3

import time

import gym
import numpy as np
import matplotlib.pyplot as plt

def display_loop():

	reward_sum_list = []
	for episode in range(600):
		print("Episode: {}".format(episode))
		reward_sum = 0
		obs = env.reset()
		for step in range(200):
			epsilon = max(1 - episode/500, .01)
			obs, reward, done, info = play_one_step(env, obs, epsilon)
			reward_sum += reward
			if done:
				break
		print("\tSum of rewards: {}".format(reward_sum))
		reward_sum_list.append(reward_sum)
		if reward_sum > 100:
			break
		if episode > 50:
			training_step(batch_size)

	plt.plot(reward_sum_list)
	plt.show()

	if end_test:
		for episode in range(5):
			obs = env.reset()
			for step in range(200):
				env.render()
				epsilon = 0
				obs, reward, done, info = play_one_step(env, obs, epsilon)
		env.close()

if __name__ == "__main__":
	training_loop(end_test=True)

