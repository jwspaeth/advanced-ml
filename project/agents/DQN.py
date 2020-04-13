
import time
from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

# 

class DQN:
    '''
    Baseline Deep Q-Network with experience replay
    '''

    def __init__(self, state_size, action_size, policy, learning_delay, loss_fn, epsilon, gamma,
        learning_rate, buffer_size, model_fn, model_param_dict, verbose=False, **kwargs):
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
        self.n_options = model_param_dict["n_options"]
        self.setup_model(model_fn, model_param_dict)

        self.epsilon_log = []
        self.reward_log = []
        self.loss_log = []
        self.deque_log = []
        self.verbose = verbose
        self.replay_buffer = {
                    "states": deque(maxlen=buffer_size),
                    "actions": deque(maxlen=buffer_size),
                    "rewards": deque(maxlen=buffer_size),
                    "next_states": deque(maxlen=buffer_size),
                    "dones": deque(maxlen=buffer_size)
        }
        self.episode = 0

    def setup_model(self, model_fn, model_param_dict):
        '''
        Compile a simple sequential model
        '''

        print("Model function: {}".format(model_fn))
        print("Model parameter dictionary: {}".format(model_param_dict))

        self.model = model_fn(**model_param_dict)
        self.model.summary()
        print("Model inputs: {}".format(self.model.inputs))
        print("Model outputs: {}".format(self.model.outputs))

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

        self.replay_buffer["states"].append(state)
        self.replay_buffer["actions"].append(action)
        self.replay_buffer["rewards"].append(reward)
        self.replay_buffer["next_states"].append(next_state)
        self.replay_buffer["dones"].append(done)

    def sample_experience_inds(self, batch_size):
        '''
        Sample batch_size number of experience indices from the replay buffer
        '''

        # If batch size greater than current length of buffer, give all indices for buffer.
        # Otherwise, get random sampling of batch_size indices.
        choice_range = len(self.replay_buffer["states"])
        if batch_size is None or batch_size > choice_range:
            indices = np.random.choice(choice_range, size=choice_range, replace=False)
        else:
            indices = np.random.choice(choice_range, size=batch_size, replace=False)

        return indices

    def sample_experience(self, inds):
        '''
        Sample experiences with indices from replay buffer
        '''

        batch = {}
        for key in self.replay_buffer.keys():
            batch[key] = [self.replay_buffer[key][index] for index in inds]


        batch["states"] = np.stack(batch["states"], axis=0)
        batch["next_states"] = np.stack(batch["next_states"], axis=0)
        batch["actions"] = np.stack(batch["actions"], axis=0)

        if len(batch["actions"].shape) == 1:
            batch["actions"] = np.expand_dims(batch["actions"], axis=1)

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

        # Get all info needed outside of gradient tape
        # Start gradient tape

        start_time = time.time()


        batch_start_time = time.time()
        # Fetch batch
        batch_inds = self.sample_experience_inds(batch_size)
        batch = self.sample_experience(batch_inds)

        learning_start_time = time.time()

        # Get number of simultaneous actions
        n_actions = len(self.model.outputs)

        # Get the next Q values from the next state
        next_Q_values_all_actions = self.get_next_Q_values(batch["next_states"])
        if type(next_Q_values_all_actions) != list:
            next_Q_values_all_actions = [next_Q_values_all_actions]

        # Take the max of all next Q value action sets
        max_next_Q_values_all_actions = [np.max(next_Q_values, axis=1) for next_Q_values in next_Q_values_all_actions]

        # Compute the target Q values
        target_Q_values_all_actions = [(batch["rewards"] + (1 - np.asarray(batch["dones"])) * self.gamma * max_next_Q_values)
                                        for max_next_Q_values in max_next_Q_values_all_actions]

        # Construct mask to hide irrelevant actions
        mask_all_actions = [tf.one_hot(batch["actions"][:, action], self.n_options[action])
                            for action in range(n_actions)]

        # Use optimizer to apply gradient to model
        with tf.GradientTape() as tape:
            loss = 0
            all_Q_values_all_actions = self.get_current_Q_values(batch["states"]) # Get all possible q values from the states

            # Aggregate loss for all actions
            for action in range(n_actions):
                masked_Q_values = all_Q_values_all_actions[action] * mask_all_actions[action] # Mask the actions which were not taken
                Q_values = tf.reduce_sum(masked_Q_values, axis=1) # Get the sum to reduce to action taken
                loss += tf.reduce_mean(self.loss_fn(target_Q_values_all_actions[action], Q_values)) # Compute the losses

            self.loss_log.append(loss) # Append to log
            grads = tape.gradient(loss, self.model.trainable_variables) # Compute the gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # Apply the gradients to the model

    def execute_episode(self, env, n_steps=None, render_flag=False, batch_size=100, verbose=False,
        train=True):
        '''
        Execute one episode, which terminates when done if flagged or step limit is reached
        '''

        # Initialize vars
        reward_total = 0
        step = 0
        done = False
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

        if verbose:
            print("\tReward: {}".format(reward_total))
        self.reward_log.append(reward_total)
        self.epsilon_log.append(self.get_epsilon())
        self.deque_log.append(len(self.replay_buffer["states"]))
        self.episode += 1

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

            if render_flag:
                env.close()
