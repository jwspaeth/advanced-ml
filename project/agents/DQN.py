import time
from collections import deque
import statistics
import json

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

import models as my_models

class DQN:
    '''
    Baseline Deep Q-Network with experience replay
    '''

    def __init__(self, state_size, action_size, policy, learning_delay, loss_fn, epsilon, gamma,
        learning_rate, buffer_size, model_config, verbose=False, reload_path=None, **kwargs):
        '''
        Initialize necessary fields
        '''
        self.type = "DQN"

        self.state_size = state_size
        self.action_size = action_size
        self.policy = policy
        self.learning_delay = learning_delay
        self.loss_fn = loss_fn
        self.epsilon = epsilon
        self.gamma = gamma
        self.optimizer = keras.optimizers.Adam(lr=learning_rate)
        self.n_options = model_config["n_options"]
        self.reload_path = reload_path
        self.model_config = model_config
        self.setup_model()

        self.epsilon_log = []
        self.reward_log = []
        self.loss_log = []
        self.deque_log = []
        self.verbose = verbose
        self.replay_buffer = deque(maxlen=buffer_size)
        self.episode = 0

    def setup_model(self):
        '''
        Compile a simple sequential model
        '''

        if self.reload_path is None:
            print("Build model from scratch")
            model_fn = self.get_model_fn(self.model_config["model_fn"])
            self.model = model_fn(**self.model_config)
        else:
            print("Reload model from file")
            with open("{}/model/model_config.json".format(self.reload_path), "r") as f:
                self.model_config = json.load(f)
            model_fn = self.get_model_fn(self.model_config["model_fn"])
            self.model = model_fn(**self.model_config)
            self.model.load_weights("{}/model/weights".format(self.reload_path))

        # Summarize model
        print("Model config: {}".format(self.model_config))
        self.model.summary()
        print("Model inputs: {}".format(self.model.inputs))
        print("Model outputs: {}".format(self.model.outputs))

    def get_model_fn(self, model_fn_name):

        print("Models attr: {}".format(my_models.__dir__()))
        try:
            model_fn = getattr(my_models, model_fn_name)
        except Exception:
            raise Exception("Error: Model function {} not found.".format(model_fn_name))

        return model_fn
        

    def get_epsilon(self):
        try:
            return self.epsilon(self.episode)
        except TypeError as e:
            return self.epsilon

    def play_one_step(self, env, state):
        '''
        Take one step in the environment based on the agent parameters
        '''
        
        action = self.policy(state, self.model, self.get_epsilon()) # Query policy

        next_state, reward, done, info = env.step(action) # Query environment
        self.memorize(state, action, reward, next_state, done) # Log
        return next_state, reward, done, info

    def memorize(self, state, action, reward, next_state, done):
        '''
        Log the experience from one step into the replay buffer as a dictionary
        '''

        state =  np.array(state, dtype=np.float32)
        next_state =  np.array(next_state, dtype=np.float32)

        self.replay_buffer.append(
                {
                    "state": state,
                    "action": action,
                    "reward": reward,
                    "next_state": next_state,
                    "done": done
                }
            )

    def sample_experience_inds(self, batch_size):
        '''
        Sample batch_size number of experience indices from the replay buffer
        '''

        # If batch size greater than current length of buffer, give all indices for buffer.
        # Otherwise, get random sampling of batch_size indices.
        choice_range = len(self.replay_buffer)
        if batch_size is None or batch_size > choice_range:
            indices = np.random.choice(choice_range, size=choice_range, replace=False)
        else:
            indices = np.random.choice(choice_range, size=batch_size, replace=False)

        return indices

    def sample_experience(self, batch_size):
        '''
        Sample experience from replay buffer

        Returns:
            inds for batch samples
            batch samples
        '''

        inds = self.sample_experience_inds(batch_size)

        batch = {}
        for key in self.replay_buffer[0].keys():
            batch[key] = [self.replay_buffer[index][key] for index in inds]


        batch["state"] = np.stack(batch["state"], axis=0)
        batch["next_state"] = np.stack(batch["next_state"], axis=0)
        batch["action"] = np.stack(batch["action"], axis=0)

        if len(batch["action"].shape) == 1:
            batch["action"] = np.expand_dims(batch["action"], axis=1)

        return batch

    def get_current_Q_values(self, states):
        return self.model(states)

    def get_next_Q_values(self, next_states):
        return self.model.predict(next_states)

    def learning_step(self, batch_size=100):
        '''
        Train the model with one batch by sampling from replay buffer
        Use the gradient tape method
        '''

        # Fetch batch
        batch = self.sample_experience(batch_size)

        # Get number of simultaneous actions
        n_actions = len(self.model.outputs)

        # Get the next Q values from the next state
        next_Q_values_all_actions = self.get_next_Q_values(batch["next_state"])
        ##print("Next Q values all actions: {}".format(next_Q_values_all_actions.shape))
        if type(next_Q_values_all_actions) != list:
            next_Q_values_all_actions = [next_Q_values_all_actions]

        # Take the max of all next Q value action sets
        max_next_Q_values_all_actions = [np.max(next_Q_values, axis=1) for next_Q_values in next_Q_values_all_actions]
        ##print("Max next Q values all actions: {}".format(max_next_Q_values_all_actions[0].shape))

        # Compute the target Q values
        target_Q_values_all_actions = [(batch["reward"] + (1 - np.asarray(batch["done"])) * self.gamma * max_next_Q_values)
                                        for max_next_Q_values in max_next_Q_values_all_actions]
        ##print("Target Q values all actions: {}".format(target_Q_values_all_actions[0].shape))

        # Construct mask to hide irrelevant actions
        mask_all_actions = [tf.one_hot(batch["action"][:, action], self.n_options[action])
                            for action in range(n_actions)]
        ##print("Mask all actions: {}".format(mask_all_actions[0].shape))

        # Use optimizer to apply gradient to model
        with tf.GradientTape() as tape:
            loss = 0
            all_Q_values_all_actions = self.get_current_Q_values(batch["state"]) # Get all possible q values from the states
            ##print("All Q values all actions: {}".format(all_Q_values_all_actions.shape))
            if type(all_Q_values_all_actions) != list:
                all_Q_values_all_actions = [all_Q_values_all_actions]
            ##print("List all Q values all actions: {}".format(len(all_Q_values_all_actions)))

            # Aggregate loss for all actions
            for action in range(n_actions):
                masked_Q_values = all_Q_values_all_actions[action] * mask_all_actions[action] # Mask the actions which were not taken
                ##print("Masked Q values: {}".format(masked_Q_values.shape))
                Q_values = tf.reduce_sum(masked_Q_values, axis=1) # Get the sum to reduce to action taken
                ##print("Q values: {}".format(Q_values.shape))
                loss += tf.reduce_mean(self.loss_fn(target_Q_values_all_actions[action], Q_values)) # Compute the losses

            self.loss_log.append(loss.numpy()) # Append to log
            grads = tape.gradient(loss, self.model.trainable_variables) # Compute the gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # Apply the gradients to the model

    def execute_episode(self, env, n_steps=None, render_flag=False, batch_size=100, verbose=False,
        train=True):
        '''
        Execute one episode, which terminates when done if flagged or step limit is reached
        '''

        # Initialize vars
        reward_total = 0 # Tracks reward sum for episode
        step = 0 # Tracks episode step
        done = False # Tracks whether episode has ended or not
        state = env.reset()
        while (n_steps is None or step < n_steps) and not done: # Continue till step count, or until done

            if render_flag: # Create visualization for environment
                env.render()

            state, reward, done, info = self.play_one_step(env, state) # Custom step function
            reward_total += reward
            step += 1
            if done:
                break

        # If train flag and episode above some threshold (to fill buffer), train
        if train and self.episode > self.learning_delay: 
            self.learning_step(batch_size=batch_size)

        # Verbose information
        if verbose:
            print("\tReward: {}".format(reward_total))

        # Log relevant data and increment episode
        self.reward_log.append(reward_total)
        self.epsilon_log.append(self.get_epsilon())
        self.deque_log.append(len(self.replay_buffer))


        # If this is the first episode or this reward total is better than any previous, set the
        #   values to their current iteration
        # Downside: One episode could have good performance as a result of a good stochastic environment
        #   reset. A better alternative would be to take a running average of performance over many episodes, and saving
        #   when that average is better than any previous average.
        if self.episode == 0 or self.best_average_reward <= self.get_100_episode_average():
            self.best_average_reward = self.get_100_episode_average()
            self.best_weights = self.model.get_weights()
            self.best_episode = self.episode
            if verbose:
                print('\tAverage reward: {}'.format(self.best_average_reward))

        # Increment episode counter for this agent
        self.episode += 1

    def get_100_episode_average(self):
        if len(self.reward_log) < 100:
            return statistics.mean(self.reward_log)
        else:
            return statistics.mean(self.reward_log[-100:])

    def execute_episodes(self, env, n_episodes, n_steps, render_flag=False, batch_size=100, verbose=False,
        train=True):
        '''
        Execute multiple episodes
        '''
        
        for episode in range(n_episodes):
            if verbose:
                print("Episode: {}".format(self.episode))

            self.execute_episode(
                env=env,
                n_steps=n_steps,
                render_flag=render_flag,
                batch_size=batch_size,
                verbose=verbose,
                train=train)

        self.model.set_weights(self.best_weights)
