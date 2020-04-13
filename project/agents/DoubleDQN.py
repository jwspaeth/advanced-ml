
import time

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np

from .TargetDQN import TargetDQN

class DoubleDQN(TargetDQN):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.type = "DoubleDQN"

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

        # Get the next online Q values from the next state 
        next_online_Q_values_all_actions = self.model.predict(batch["next_states"])
        if type(next_online_Q_values_all_actions) != list:
            next_online_Q_values_all_actions = [next_online_Q_values_all_actions]

        # Choose the argmaxes from the online model
        next_Q_values_argmaxes = [np.argmax(next_online_Q, axis=1) for next_online_Q in next_online_Q_values_all_actions]

        # Create masks for the best online argmaxes
        all_next_masks = [tf.one_hot(next_Q_argmax, self.n_options[i]).numpy()
            for i, next_Q_argmax in enumerate(next_Q_values_argmaxes)]

        next_target_Q_values_all_actions = self.target_model.predict(batch["next_states"])
        if type(next_target_Q_values_all_actions) != list:
            next_target_Q_values_all_actions = [next_target_Q_values_all_actions]

        # Get the value of the best action chosen by online model
        max_next_Q_values_all_actions = [ (next_target_Q_values * next_masks).sum(axis=1)
            for next_target_Q_values, next_masks in zip(next_target_Q_values_all_actions, all_next_masks)]

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

            self.loss_log.append(loss.numpy()) # Append to log
            grads = tape.gradient(loss, self.model.trainable_variables) # Compute the gradients
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables)) # Apply the gradients to the model

