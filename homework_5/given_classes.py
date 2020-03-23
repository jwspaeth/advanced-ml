import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt
import tensorflow.keras as keras

#from tensorflow import keras
from tensorflow.keras.layers import LeakyReLU, UpSampling1D, Input, InputLayer, Reshape, Activation, Lambda, AveragePooling1D
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Conv2DTranspose, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import RMSprop
import random
#import skimage.transform as sktr
import gym
from mpl_toolkits.mplot3d import Axes3D
import re
 

#from sklearn.p
import sklearn.metrics

from sklearn.utils.extmath import cartesian

class numpyBuffer:
    '''
    Circular buffer using a numpy array
    
    In this case, we only append to this buffer and overwrite values once we wrap-around
    '''
    def __init__(self, maxsize=100, ndims=1, dtype=np.float32):
        '''
        Constructor for the buffer
        
        :param maxsize: Maximum number of rows that can be stored in the buffer
        :param ndims: The number of columns in the buffer       
        '''
        
        self.buffer = np.zeros((maxsize,ndims), dtype=dtype)
        self.maxsize=maxsize
        self.ndims=ndims
        self.back = 0
        self.full = False
    
    def size(self):
        '''
        :return: The number of items stored in the buffer
        '''
        if(self.full):
            return self.maxsize
        else:
            return self.back
        
    def append(self, rowvec):
        '''
        Append a row to the buffer
        
        :param rowvec: Numpy row vector of values to append.  Must be 1xndims
        '''
        self.buffer[self.back,:] = rowvec
        self.back = self.back+1
        if self.back >= self.maxsize:
            self.back = 0
            self.full = True
            
    def getrows(self, row_indices):
        '''
        Return a set of indicated rows
        
        :param row_indices: Array of row indices into the buffer
        :return: len(row_indices)xndims numpy array
        '''
        return self.buffer[row_indices,:]



class myAgent:
    def __init__(self, state_size, action_size, action_continuous=None, epsilon=.01, gamma=0.99, 
                 lrate=.001, maxlen=10000):
        '''
        :param state_size: Number of state variables
        :param action_size: Number of actions (will use one-hot encoded actions)
        :param action_continuous: List of continuous actions that correspond to the discrete choices.  If None, then
                we have a built-in set of discrete actions
        :param epsilon: Constant exploration rate
        :param gamma: Constant discout rate
        :param lrate: Learning rate
        :param action_discrete: Network produces one Q-value for each discrete action 
                (True is the only supported case)
        :param maxlen: Maximum length of the circular experience buffer
        
        Experience buffer is designed for quick access to prior experience
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.action_continuous = action_continuous
        self.epsilon=epsilon
        self.gamma=gamma
        self.reward_log = []
        self.verbose = False
        self.verbose_execute = False
        self.lrate=lrate
        self.action_discrete=(action_continuous == None)
        self.log_observation = numpyBuffer(maxlen, state_size)
        self.log_observation_new = numpyBuffer(maxlen, state_size)
        self.log_action = numpyBuffer(maxlen, 1, dtype=np.int16)
        self.log_reward = numpyBuffer(maxlen, 1)
        self.log_done = numpyBuffer(maxlen, 1, dtype=np.bool)
        
    def build_model_predictor(self, n_units, activation='elu', lambda_regularization=None):
        '''
        Simple sequential model.
        
        :param n_units: Number of units in each hidden layer (a list)
        :param activation: Activation function for the hidden units
        :param lambda_regularization: None or a continuous value (currently not used)
        '''
        model = Sequential()
        i = 0
        
        # Input layer
        model.add(InputLayer(input_shape=(self.state_size,)))
        
        # Loop over hidden layers
        for n in n_units:
            model.add(Dense(n, 
                        activation=activation,
                        use_bias=True,
                        kernel_initializer='truncated_normal', 
                        bias_initializer='zeros', 
                        name = "D"+str(i)))
                        #kernel_regularizer=keras.regularizers.l2(lambda_regularization),
                        #bias_regularizer=keras.regularizers.l2(lambda_regularization)))
            i=i+1
            
        # model.add(BatchNormalization())
        # Output layer
        model.add(Dense(self.action_size, 
                        activation=None,
                        use_bias=True,
                        kernel_initializer='truncated_normal', 
                        bias_initializer='zeros',  
                        name = "D"+str(i)))
                        #kernel_regularizer=keras.regularizers.l2(lambda_regularization),
                        #bias_regularizer=keras.regularizers.l2(lambda_regularization)))
        
        return model
        
        
    def build_model(self, n_units, activation='elu', lambda_regularization=None):
        '''
        Simple sequential model.
        
        :param n_units: Number of units in each hidden layer (a list)
        :param activation: Activation function for the hidden units
        :param lambda_regularization: None or a continuous value (currently not used)
        '''
        model = self.build_model_predictor(n_units=n_units, activation=activation, lambda_regularization=lambda_regularization)
        self.model = model
        
        # We do not have a separate model for learning
        self.model_learning = None
        
        # Configure model
        opt = keras.optimizers.Adam(lr=self.lrate, beta_1=0.9, beta_2=0.999, 
                            epsilon=None, decay=0.0, amsgrad=False)
        
        model.compile(loss='mse', optimizer=opt)
        
        print(model.summary())
        
    def choose_action(self, observation, verbose=False):
        '''
        epsilon-greedy choice of discrete action
        
        :returns: (discrete_action, explore_bit)

        '''
        if(np.random.rand() <= self.epsilon):
            return np.random.randint(self.action_size), True
        else:
            pred = self.model.predict(observation)[0]
            if verbose:
                print(pred)
            return np.argmax(pred), False
    
    def choose_action_continuous(self, observation, verbose=False):
        '''
        epsilon-greedy choice of continuous action
        
        :returns: (discrete_action, continuous_action, explore_bit)
        '''
        action_index, explore = self.choose_action(observation, verbose)
        return action_index, self.action_continuous[action_index], explore
    
    def log_experience(self, observation, action_index, reward, observation_new, done):
        ''' 
        Store the last step in the circular buffer
        '''
        observation =  np.array(observation, ndmin=2)
        observation_new =  np.array(observation_new, ndmin=2)
        
        self.log_observation.append(observation)
        self.log_observation_new.append(observation_new)
        self.log_action.append(action_index)
        self.log_reward.append(reward)
        self.log_done.append(done)
                
    def learning_step(self, batch_size=200):
        '''
        Iterate over a minibatch of the stored experience & take a learning step with each

        :param batch_size: Size of the batch to do learning with
        
        '''
        
        # Sample from the prior experience.  How we do this depends on how much
        #  experience that we have accumulated so far
        if self.log_observation.size() < batch_size:
            minibatch_inds = range(self.log_observation.size())
            #return
        else:
            # Random sample from the buffer
            minibatch_inds = random.sample(range(self.log_observation.size()), batch_size)
        
        print("Creating batch:", len(minibatch_inds))
        
        # Observations for the samples
        observations = self.log_observation.getrows(minibatch_inds)
        # Q values for the samples
        targets = ???
        # Next observations
        observations_new = self.log_observation_new.getrows(minibatch_inds)
        # Next Q value
        q_next = ????
        # Max next Q value
        q_next_max = np.max(q_next, axis=1)
        
        # Get the rewards received for the samples in the minibatch
        rewards = self.log_reward.getrows(minibatch_inds)[:,0]

        # For each sample, was it the last step?
        dones = self.log_done.getrows(minibatch_inds)[:,0]  
        # Samples that were done (list of indices)
        done_list = ????
        # Samples that were not done (list of indices)
        done_not_list = ????
        
        # Get the actions taken for the minibatch
        actions = self.log_action.getrows(minibatch_inds)[:,0]
        
        # Update targets: for each example, only one action is updated
        #  (the one that was actually executed)
        
        # Last step in the episodes
        targets[done_list, actions[done_list]] = ????
        # Other steps
        targets[done_not_list, actions[done_not_list]] = ???
        
        # Update the Q-function
        self.model.fit(observations, targets, epochs=1, verbose=0)

        if self.verbose:
            print(observations, targets)
    
    def execute_trial(self, env, nsteps, render_flag=False, batch_size=100):
        '''
        A trial terminates at nsteps or when the environment says we must stop.
        
        '''
        observation = env.reset()
        observation = np.array(observation, ndmin=2)
        # Accumulator for total reward
        reward_total = 0
        
        # Loop over each step
        for i in range(nsteps):
            if render_flag:
                env.render()
                
            # Some environments require discrete actions, while others require continous actions
            if self.action_discrete:
                action_index, explore = self.choose_action(observation, verbose=self.verbose_execute)
                observation_new, reward, done, info = env.step(action_index) 
            else:
                # Figure out which action to execute
                action_index, action_continuous, explore = self.choose_action_continuous(observation, verbose=self.verbose_execute)
                observation_new, reward, done, info = env.step(action_continuous)
                
            # Remember reward
            reward_total = reward_total + reward
            if self.verbose_execute:
                print(observation, action_index, reward, observation_new, done)
                
            # Log this step 
            #if not explore:
            self.log_experience(observation, action_index, reward, 
                                    observation_new, done)
                
            if done:
                # Environment says we are done
                break
                
            # Prepare for the next step
            observation = observation_new
            observation = np.array(observation, ndmin=2)
            
        # Learning
        #print("before learning")
        self.learning_step(batch_size=batch_size)
        if render_flag:
            env.close()
        print(reward_total)
        
        # Log accumulated reward for this trial
        self.reward_log.append(reward_total)
        
    def execute_ntrials(self, env, ntrials, nsteps, render_flag=False, batch_size=100):
        '''
        Execute the specified number of trials
        '''
        for _ in range(ntrials):
            self.execute_trial(env, nsteps, render_flag, batch_size)
        
           