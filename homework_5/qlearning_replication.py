#!/usr/bin/env python3

import time
from collections import deque

import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.layers import LeakyReLU, UpSampling1D, Input, InputLayer, Reshape, Activation, Lambda, AveragePooling1D
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop

# Try different updates for target model

# Best: n_units [32, 16]; steps None; 1000 episodes; .95 gamma; buffer size None; batch size 64; epsilon decay 1->.01;
#       target_update 25

def epsilon_greedy_policy(state, model, epsilon = 0):
    '''
    Simple policy which either gets the next action or random action with chance epsilon
    '''
    if np.random.rand() < epsilon:
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def random_policy(*args):
    return np.random.randint(2)

def epsilon_episode_decay(initial_epsilon, min_epsilon):

        # Decay epsilon to some min value according to episode
        return lambda episode: max(initial_epsilon - episode / 500, min_epsilon)


class MyAgent:
    '''
    Manages RL process
    '''

    def __init__(self, state_size, action_size, policy, loss_fn, 
                epsilon = 0, gamma = 0.99, 
                lrate = .001, maxlen = 2000):
        '''
        Initialize necessary fields
        '''

        self.state_size = state_size
        self.action_size = action_size
        self.policy = policy
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.epsilon_log = []
        self.reward_log = []
        self.loss_log = []
        self.deque_log = []
        self.verbose = False
        self.verbose_execute = False
        self.lrate = lrate
        self.replay_buffer = {
                    "states": deque(maxlen=maxlen),
                    "actions": deque(maxlen=maxlen),
                    "rewards": deque(maxlen=maxlen),
                    "next_states": deque(maxlen=maxlen),
                    "dones": deque(maxlen=maxlen)
        }
        self.optimizer = keras.optimizers.Adam(lr=lrate)
        self.episode = 0

    def build_model(self, n_units, activation='elu', l2_lambda=.01):
        '''
        Build a simple sequential model.
        '''

        model = Sequential()
        i = 0
        
        # Input layer
        model.add(InputLayer(input_shape=(self.state_size,)))
        
        # Loop over hidden layers
        for n in n_units:
            model.add(Dense(n, 
                        activation=activation,
                        name = "D"+str(i)))
            i=i+1
            
        # model.add(BatchNormalization())
        # Output layer
        model.add(Dense(self.action_size, 
                        activation=None,
                        name = "D"+str(i)))
        
        return model

    def compile_model(self, n_units, activation='elu'):
        '''
        Compile a simple sequential model
        '''

        model = self.build_model(n_units=n_units, activation=activation)
        self.model = model
        
        # Configure model
        opt = keras.optimizers.Adam(lr=self.lrate)
        model.compile(loss='mse', optimizer=opt)
        
        print(model.summary())

    def setup_model(self, n_units, activation='elu'):
        '''
        Compile a simple sequential model
        '''

        model = self.build_model(n_units=n_units, activation=activation)
        self.model = model
        model.summary()

    def get_epsilon(self):
        try:
            return self.epsilon(self.episode)
        except TypeError as e:
            return self.epsilon

    def step(self, env, state):
        '''
        Take one step in the environment based on the agent parameters
        '''
        
        action = self.policy(state, self.model, self.get_epsilon()) # Query policy

        next_state, reward, done, info = env.step(action) # Query environment
        self.log_experience(state, action, reward, next_state, done) # Log
        return next_state, reward, done, info

    def log_experience(self, state, action, reward, next_state, done):
        '''
        Log the experience from one step into the replay buffer as a dictionary
        '''

        state =  np.array(state, ndmin=2)
        next_state =  np.array(next_state, ndmin=2)
        
        self.replay_buffer["states"].append(state)
        self.replay_buffer["actions"].append(action)
        self.replay_buffer["rewards"].append(reward)
        self.replay_buffer["next_states"].append(next_state)
        self.replay_buffer["dones"].append(done)

    def sample_experience(self, batch_size):
        '''
        Sample batch_size number of experience from the replay buffer
        '''

        # If batch size greater than current length of buffer, give all indices for buffer.
        # Otherwise, get random sampling of batch_size indices.
        if batch_size > len(self.replay_buffer["states"]):
            indices = list(range(len(self.replay_buffer["states"])))
        else:
            indices = np.random.randint(len(self.replay_buffer["states"]), size=batch_size)

        batch = {}
        for key in self.replay_buffer.keys():
            batch[key] = [self.replay_buffer[key][index] for index in indices]

        return batch

    def learning_step(self, batch_size=100):
        '''
        Train the model with one batch by sampling from replay buffer
        '''

        # Fetch batch
        batch = self.sample_experience(batch_size)

        # Create target q values, with mask to disregard irrelevant actions
        next_Q_values = self.model.predict(batch["next_states"])
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (batch["rewards"] + (1 - np.asarray(batch["dones"])) * self.gamma * max_next_Q_values)
        mask = tf.one_hot(batch["actions"], self.action_size)

        # Use optimizer to apply gradient to model
        with tf.GradientTape() as tape:
            all_Q_values = self.model(batch["states"]) # Get all possible q values from the states
            masked_Q_values = all_Q_values * mask # Mask the actions which were not taken
            Q_values = tf.reduce_sum(masked_Q_values, axis=1, keepdims=True) # Get the sum along each action column
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values)) # Compute the losses
            self.loss_log.append(loss)
            grads = tape.gradient(loss, self.model.trainable_variables) # Compute the gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # Apply the gradients to the model

    def get_reward_log_average(self):
        
        avg = 0
        for total in self.reward_log:
            avg += total
        avg /= len(self.reward_log)

        return avg

    def execute_episode(self, env, n_steps, n_learning_steps=1, 
        render_flag=False, batch_size=100, verbose=False, train=True, test=False):
        '''
        Execute one episode, which terminates when done if flagged or step limit is reached
        '''

        # Determine whether or not to continue forever
        if n_steps == None:
            infinite_flag = True
        else:
            infinite_flag = False

        reward_total = 0
        step = 0
        if infinite_flag: # If n_steps isn't defined, run until the environment is finished
            n_steps = step + 1
        done = False
        state = env.reset()
        while step < n_steps and not done:

            if render_flag:
                env.render()

            state, reward, done, info = self.step(env, state)
            reward_total += reward
            step += 1
            if infinite_flag:
                n_steps = step + 1
            if done and not test:
                break

        if train and self.episode > 50:
            for i in range(n_learning_steps):
                self.learning_step(batch_size=batch_size)

        self.reward_log.append(reward_total)
        self.epsilon_log.append(self.get_epsilon())
        self.deque_log.append(len(self.replay_buffer["states"]))
        self.episode += 1

    def execute_episodes(self, env, n_episodes, n_steps, n_learning_steps=1, 
        render_flag=False, batch_size=100, verbose=False, train=True, test=False):
        '''
        Execute multiple episodes
        '''
        
        for episode in range(n_episodes):
            if verbose:
                print("Episode: {}".format(episode))

            self.execute_episode(env, n_steps, n_learning_steps=n_learning_steps,
                render_flag=render_flag, batch_size=batch_size, verbose=verbose, train=train,
                test=test)

            if render_flag:
                env.close()

        return self.get_reward_log_average()

class MyAgentTarget(MyAgent):
    '''
    Uses fixed q value targets
    '''

    def __init__(self, state_size, action_size, policy, loss_fn, 
                epsilon = 0, gamma = 0.99, 
                lrate = .001, maxlen = 2000):
        super().__init__(state_size, action_size, policy, loss_fn, epsilon, gamma, lrate, maxlen)

        self.state_size = state_size
        self.action_size = action_size
        self.policy = policy
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.gamma = gamma
        self.model = None
        self.epsilon_log = []
        self.reward_log = []
        self.loss_log = []
        self.deque_log = []
        self.verbose = False
        self.verbose_execute = False
        self.lrate = lrate
        self.replay_buffer = {
                    "states": deque(maxlen=maxlen),
                    "actions": deque(maxlen=maxlen),
                    "rewards": deque(maxlen=maxlen),
                    "next_states": deque(maxlen=maxlen),
                    "dones": deque(maxlen=maxlen)
        }
        self.optimizer = keras.optimizers.Adam(lr=lrate)
        self.episode = 0

    def setup_model(self, n_units, activation='elu'):
        '''
        Compile a simple sequential model
        '''

        model = self.build_model(n_units=n_units, activation=activation)
        self.model = model
        model.summary()

        self.create_target_model()

    def create_target_model(self):
        self.target_model = keras.models.clone_model(self.model)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        print("\tUpdate target model")

    def learning_step(self, batch_size=100):
        '''
        Train the model with one batch by sampling from replay buffer
        '''

        # Fetch batch
        batch = self.sample_experience(batch_size)

        # Create target q values, with mask to disregard irrelevant actions
        next_Q_values = self.target_model.predict(batch["next_states"])
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (batch["rewards"] + (1 - np.asarray(batch["dones"])) * self.gamma * max_next_Q_values)
        mask = tf.one_hot(batch["actions"], self.action_size)

        # Use optimizer to apply gradient to model
        with tf.GradientTape() as tape:
            all_Q_values = self.model(batch["states"]) # Get all possible q values from the states
            masked_Q_values = all_Q_values * mask # Mask the actions which were not taken
            Q_values = tf.reduce_sum(masked_Q_values, axis=1, keepdims=True) # Get the sum along each action column
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values)) # Compute the losses
            self.loss_log.append(loss)
            grads = tape.gradient(loss, self.model.trainable_variables) # Compute the gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # Apply the gradients to the model

    def execute_episode(self, env, n_steps, n_learning_steps=1, render_flag=False,
        batch_size=100, verbose=False, train=True, test=False):
        '''
        Execute one episode, which terminates when done if flagged or step limit is reached
        '''

        # Determine whether or not to continue forever
        if n_steps == None:
            infinite_flag = True
        else:
            infinite_flag = False

        reward_total = 0
        step = 0
        if infinite_flag: # If n_steps isn't defined, run until the environment is finished
            n_steps = step + 1
        done = False
        state = env.reset()
        while step < n_steps and not done:

            if render_flag:
                env.render()

            state, reward, done, info = self.step(env, state)
            reward_total += reward
            step += 1
            if infinite_flag:
                n_steps = step + 1
            if done and not test:
                break

        if train and self.episode > 50:

            for i in range(n_learning_steps):
                self.learning_step(batch_size=batch_size)
            if self.episode % 25 == 0:
                self.update_target_model()

        if verbose:
            print("\tReward total: {}".format(reward_total))
        self.reward_log.append(reward_total)
        self.epsilon_log.append(self.get_epsilon())
        self.deque_log.append(len(self.replay_buffer["states"]))
        self.episode += 1


def runDeepAgent(n_units, env, n_silent_episodes, n_visible_episodes, n_steps, n_learning_steps=1):
    '''
    Deep agent section
    '''
    # Cart-pole is a discrete action environment (provided continous values are dummies)
    deepAgent = MyAgent(
        state_size = env.observation_space.shape[0],
        action_size = env.action_space.n,
        policy = epsilon_greedy_policy,
        loss_fn = keras.losses.mean_squared_error,
        gamma = 0.99,
        #epsilon = .1,
        epsilon = epsilon_episode_decay(1, .01),
        lrate = .001,
        maxlen = 100000)
    deepAgent.setup_model(n_units=n_units)

    print("Deep agent for {} silent episodes...".format(n_silent_episodes))
    # Execute n trials silently
    deepAgent.execute_episodes(
                        env = env,
                        n_episodes = n_silent_episodes,
                        n_steps = n_steps,
                        n_learning_steps = n_learning_steps,
                        render_flag = False,
                        batch_size = 64,
                        verbose = True
                        )

    print("Deep agent for {} live episodes...".format(n_visible_episodes))
    # Execute trials while rendering
    deepAgent.execute_episodes(
                        env = env,
                        n_episodes = n_visible_episodes,
                        n_steps = n_steps,
                        n_learning_steps = n_learning_steps,
                        render_flag = True,
                        batch_size = 64,
                        verbose = False,
                        test=True
                        )
    return deepAgent

def runDeepAgentTarget(n_units, env, n_silent_episodes, n_visible_episodes, n_steps, n_learning_steps=1):
    '''
    Deep agent section
    '''
    # Cart-pole is a discrete action environment (provided continous values are dummies)
    deepAgent = MyAgentTarget(
        state_size = env.observation_space.shape[0],
        action_size = env.action_space.n,
        policy = epsilon_greedy_policy,
        loss_fn = keras.losses.mean_squared_error,
        gamma = 0.95,
        #epsilon = .1,
        epsilon = epsilon_episode_decay(1, .01),
        lrate = .001,
        maxlen = 100000)
    deepAgent.setup_model(n_units=n_units)

    print("Deep agent target for {} silent episodes...".format(n_silent_episodes))
    # Execute n trials silently
    deepAgent.execute_episodes(
                        env = env,
                        n_episodes = n_silent_episodes,
                        n_steps = n_steps,
                        n_learning_steps = n_learning_steps,
                        render_flag = False,
                        batch_size = 64,
                        verbose = True
                        )

    print("Deep agent target for {} live episodes...".format(n_visible_episodes))
    # Execute trials while rendering
    deepAgent.execute_episodes(
                        env = env,
                        n_episodes = n_visible_episodes,
                        n_steps = n_steps,
                        n_learning_steps = n_learning_steps,
                        render_flag = True,
                        batch_size = 64,
                        verbose = False,
                        test=True
                        )
    return deepAgent

def runRandomAgent(env, n_silent_episodes, n_visible_episodes, n_steps):
    '''
    Random agent section
    '''
    ranAgent = MyAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        policy=random_policy,
        loss_fn=keras.losses.mean_squared_error,
        )

    print("Random agent for {} silent episodes...".format(n_silent_episodes))
    # Execute n trials silently
    ranAgent.execute_episodes(
                        env=env,
                        n_episodes=n_silent_episodes,
                        n_steps=n_steps,
                        render_flag=False,
                        verbose=True,
                        train=False
                        )
    print("Random agent for {} live episodes...".format(n_visible_episodes))
    ranAgent.execute_episodes(
                        env=env,
                        n_episodes=n_visible_episodes,
                        n_steps=n_steps,
                        render_flag=True,
                        verbose=False,
                        train=False,
                        test=True
                        )
    return ranAgent

def main():

    env = gym.make('CartPole-v1')

    print("Setup agent...")
    print("\tState space: {}".format(env.observation_space.shape))
    print("\tAction space: {}".format(env.action_space.n))

    n_learning_steps = 5
    silent_episodes = 1000
    visible_episodes = 5
    steps = None
    print("Number of learning steps: {}".format(n_learning_steps))
    print("Number of steps: {}".format(steps))

    #deepAgent1 = runDeepAgent([24, 48], env, silent_episodes, visible_episodes, steps)
    deepAgent2 = runDeepAgentTarget([32, 16], env, silent_episodes, visible_episodes, steps, n_learning_steps)

    '''
    deepAgent3 = runDeepAgent([32,], env, silent_episodes, visible_episodes, steps)
    deepAgent4 = runDeepAgentTarget([32], env, silent_episodes, visible_episodes, steps)
    '''

    print("Plot!")
    # Show accumulated reward as a function of trial

    fig, axs = plt.subplots(2, 2)
    
    axs[0, 0].plot(deepAgent2.reward_log, label="Deep Agent Target -- Avg: {:.2f}".format(deepAgent2.get_reward_log_average()))
    axs[0, 0].set_ylim([0, 200])
    axs[0, 0].legend()
    
    axs[0, 1].plot(deepAgent2.deque_log, label="Deque Size")
    axs[0, 1].legend()

    '''
    axs[1, 0].plot(deepAgent3.reward_log, label="Deep Agent 3 -- Avg: {:.2f}".format(deepAgent3.get_reward_log_average()))
    axs[1, 0].set_ylim([0, 200])
    axs[1, 0].legend()

    axs[1, 1].plot(deepAgent4.reward_log, label="Deep Agent 4 -- Avg: {:.2f}".format(deepAgent4.get_reward_log_average()))
    axs[1, 1].set_ylim([0, 200])
    axs[1, 1].legend()
    '''

    plt.show()

if __name__ == "__main__":
    main()




