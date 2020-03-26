#!/usr/bin/env python3

import time

import gym
import tensorflow as tf
from tensorflow import keras
import numpy as np

'''
Algorithm
	Play episode
	During episode, compute gradient for each action
	After episode, compute advantage for each action
	If advantage for an action is positive, apply the gradient to make
		it more likely. If advantage for an action is negative, apply
		the opposite of the gradient to make it less likely
'''

def basic_policy(obs):
	angle = obs[2]
	return 0 if angle < 0 else 1

def play_one_step(env, obs, model, loss_fn):
	with tf.GradientTape() as tape:
		left_proba = model(obs[np.newaxis])
		action = (tf.random.uniform([1, 1]) > left_proba)

		y_target = tf.constant([[1.]] - tf.cast(action, tf.float32))
		loss = tf.reduce_mean(loss_fn(y_target, left_proba))
		grads = tape.gradient(loss, model.trainable_variables)

		obs, reward, done, info = env.step(int(action[0, 0].numpy()))

		return obs, reward, done, grads

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn, graphics=False):
	all_rewards = []
	all_grads = []
	for episode in range(n_episodes):
		current_rewards = []
		current_grads = []
		obs = env.reset()
		for step in range(n_max_steps):
			if graphics:
				env.render()
			obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
			current_rewards.append(reward)
			current_grads.append(grads)
			if done:
				break
		if graphics:
			env.close()
		all_rewards.append(current_rewards)
		all_grads.append(current_grads)

	return all_rewards, all_grads

def discount_rewards(rewards, discount_factor):
	discounted = np.array(rewards)
	for step in range(len(rewards) - 2, -1 , -1):
		discounted[step] += discounted[step + 1] * discount_factor

	return discounted

def discount_and_normalize_rewards(all_rewards, discount_factor):
	all_discounted_rewards = [discount_rewards(rewards, discount_factor) for rewards in all_rewards]
	flat_rewards = np.concatenate(all_discounted_rewards)
	reward_mean = flat_rewards.mean()
	reward_std = flat_rewards.std()

	return [(discounted_rewards - reward_mean) / reward_std for discounted_rewards in all_discounted_rewards]

def training_loop(n_iterations, n_episodes_per_update, n_max_steps, discount_factor,
	end_test=False):

	env = gym.make("CartPole-v1")

	n_inputs = 4
	model = keras.models.Sequential([
			keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
			keras.layers.Dense(1, activation="sigmoid"),
		])

	optimizer = keras.optimizers.Adam(lr=.01)
	loss_fn = keras.losses.binary_crossentropy

	for iteration in range(n_iterations):
		print("Iteration: {}".format(iteration))
		all_rewards, all_grads = play_multiple_episodes(
				env = env,
				n_episodes = n_episodes_per_update,
				n_max_steps = n_max_steps,
				model = model,
				loss_fn = loss_fn
			)

		all_final_rewards = discount_and_normalize_rewards(all_rewards, discount_factor)

		all_mean_grads = []
		for var_index in range(len(model.trainable_variables)):
			mean_grads = tf.reduce_mean(
					[final_reward * all_grads[episode_index][step][var_index]
						for episode_index, final_rewards in enumerate(all_final_rewards)
							for step, final_reward in enumerate(final_rewards)], axis=0)
			all_mean_grads.append(mean_grads)

		optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

	if end_test:
		play_multiple_episodes(
				env = env,
				n_episodes = 2,
				n_max_steps = n_max_steps,
				model = model,
				loss_fn = loss_fn,
				graphics = True
			)

if __name__ == "__main__":

	n_iterations = 150
	n_episodes_per_update = 10
	n_max_steps = 200
	discount_factor = .95
	training_loop(
			n_iterations = n_iterations,
			n_episodes_per_update = n_episodes_per_update,
			n_max_steps = n_max_steps,
			discount_factor = discount_factor,
			end_test = True
		)







