
import tensorflow.keras as keras

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense

from .DQN import DQN

class CuriosityDQN(DQN):

    def __init__(self, curiosity_n_units, curiosity_weight=1, **kwargs):
        super().__init__(**kwargs)

        self.type = "CuriosityDQN"
        self.curiosity_optimizer = keras.optimizers.Adam(lr=kwargs["learning_rate"])
        self.curiosity_weight = curiosity_weight
        self.curiosity_log = []
        self.curiosity_reward_log = []
        self.raw_score_log = []
        self.q_val_log = []

        self.setup_curiosity_model(curiosity_n_units)

    def setup_curiosity_model(self, n_units, activation="elu"):
        '''
        Build a simple sequential model.
        '''

        model = Sequential()
        i = 0
        
        # Input layer
        model.add(InputLayer(input_shape=(self.state_size+self.action_size,)))
        
        # Loop over hidden layers
        for n in n_units:
            model.add(Dense(n, 
                        activation=activation,
                        name = "D"+str(i)))
            i=i+1
            
        # model.add(BatchNormalization())
        # Output layer
        model.add(Dense(self.state_size+self.action_size, 
                        activation=None,
                        name = "D"+str(i)))
        
        self.curiosity_model = model
        self.curiosity_model.summary()

    def curiosity_policy(self, state):

        actions = [-1, 0, 1]
        Q_values = self.model.predict(state[np.newaxis])
        curiosity_scores = []
        for action in actions:
            curiosity_scores.append(self.get_curiosity_score(state, action))
        curiosity_scores = np.asarray(curiosity_scores)
        raw_scores = Q_values[0] + curiosity_scores

        best_index = np.argmax(raw_scores)
        self.raw_score_log.append(raw_scores[best_index])
        self.curiosity_reward_log.append(curiosity_scores[best_index])
        self.q_val_log.append(Q_values[0][best_index])
        return np.argmax(raw_scores)

    def play_one_step(self, env, state):
        '''
        Take one step in the environment based on the agent parameters
        '''

        action = self.curiosity_policy(state) # Query policy

        next_state, reward, done, info = env.step(action) # Query environment
        self.memorize(state, action, reward, next_state, done) # Log

        return next_state, reward, done, info

    def get_curiosity_score(self, state, action):
        action_encoding = tf.one_hot(action, self.action_size)
        action_encoding = tf.expand_dims(action_encoding, axis=0)
        state = np.expand_dims(state, axis=0)

        ins = np.concatenate((state, action_encoding), axis=1)
        outs = self.curiosity_model.predict(ins)
        loss = np.mean(self.loss_fn(ins, outs))

        return loss

    def update_curiosity_model(self, batch_size):
        '''
        Train the model with one batch by sampling from replay buffer
        Use the gradient tape method
        '''

        # Fetch batch
        batch_inds = self.sample_experience_inds(batch_size)
        batch = self.sample_experience(batch_inds)
        batch["actions"] = np.stack(batch["actions"], axis=0)
        action_encoding = tf.one_hot(batch["actions"], self.action_size)

        # Create target q values, with mask to disregard irrelevant actions
        ins = np.concatenate((batch["states"], action_encoding), axis=1)

        # Use optimizer to apply gradient to model
        with tf.GradientTape() as tape:
            outs = self.curiosity_model(ins)
            loss = tf.reduce_mean(self.loss_fn(ins, outs))
            print("\tCuriosity loss: {}".format(loss))
            self.curiosity_log.append(loss)
            grads = tape.gradient(loss, self.curiosity_model.trainable_variables)
            self.curiosity_optimizer.apply_gradients(zip(grads, self.curiosity_model.trainable_variables))

    def execute_episode(self, **kwargs):
        super().execute_episode(**kwargs)

        self.update_curiosity_model(kwargs["batch_size"])