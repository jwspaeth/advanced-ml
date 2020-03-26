#!/usr/bin/env python3

import time
from collections import deque

import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

batch_size = 32
discount_factor = .95
optimizer = keras.optimizers.Adam(lr=.001)
loss_fn = keras.losses.mean_squared_error

env = gym.make("CartPole-v0")

input_shape = [4]
n_outputs = 2

model = keras.models.Sequential([
		keras.layers.Dense(32, activation="elu", input_shape=input_shape),
		keras.layers.Dense(32, activation="elu"),
		keras.layers.Dense(n_outputs)
	])

replay_buffer = deque(maxlen=2000)

def training_loop(end_test=False):

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

def training_step(batch_size):
	experiences = sample_experiences(batch_size)
	states, actions, rewards, next_states, dones = experiences
	next_Q_values = model.predict(next_states)
	max_next_Q_values = np.max(next_Q_values, axis=1)
	target_Q_values = (rewards + 
						(1 - dones) * discount_factor * max_next_Q_values)
	mask = tf.one_hot(actions, n_outputs)
	with tf.GradientTape() as tape:
		all_Q_values = model(states)
		Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
		loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
		grads = tape.gradient(loss, model.trainable_variables)
		optimizer.apply_gradients(zip(grads, model.trainable_variables))

def play_one_step(env, state, epsilon):
	action = epsilon_greedy_policy(state, epsilon)
	next_state, reward, done, info = env.step(action)
	replay_buffer.append((state, action, reward, next_state, done))
	return next_state, reward, done, info

def sample_experiences(batch_size):
	indices = np.random.randint(len(replay_buffer), size=batch_size)
	batch = [replay_buffer[index] for index in indices]
	states, actions, rewards, next_states, dones = [
		np.array([experience[field_index] for experience in batch])
		for field_index in range(5)]
	return states, actions, rewards, next_states, dones

def epsilon_greedy_policy(state, epsilon=0):
	if np.random.rand() < epsilon:
		return np.random.randint(2)
	else:
		Q_values = model.predict(state[np.newaxis])
		return np.argmax(Q_values[0])

if __name__ == "__main__":
	training_loop(end_test=True)

