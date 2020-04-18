#!/usr/bin/env python3

import gym

from .agents import DQN

def setup_cartpole():
	return gym.make('CartPole-v1')

def setup_dqn():
	pass

def setup():
	return setup_cartpole(), set_dqn()

