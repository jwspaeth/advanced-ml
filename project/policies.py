
import numpy as np

def epsilon_greedy_policy_lunar_lander(state, model, epsilon = 0):

    if np.random.rand() < epsilon:
        return np.random.randint(0, high=4)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])

def epsilon_greedy_policy_generator(action_low, action_high_plus):

    def epsilon_greedy_policy(state, model, epsilon = 0):
        '''
        Simple policy which either gets the next action or random action with chance epsilon in cartpole
        '''

        if np.random.rand() < epsilon:
            return np.random.randint(action_low, high=action_high_plus)
        else:
            Q_values = model.predict(state[np.newaxis])
            return np.argmax(Q_values[0])

    return epsilon_greedy_policy

def epsilon_greedy_policy_car_generator(low_high_list):

    def epsilon_greedy_policy(state, model, epsilon = 0):
        '''
        Simple policy which either gets the next action or random action with chance epsilon in cartpole
        '''

        if np.random.rand() < epsilon:
            actions = [np.random.randint(low_high[0], high=low_high[1]) for low_high in low_high_list]
            return actions
        else:
            Q_values = model.predict(state[np.newaxis])
            actions = [np.argmax(np.squeeze(Q_val)) for Q_val in Q_values]
            return actions

    return epsilon_greedy_policy


def random_policy(*args):
    '''
    Randomly chooses an action in cartpole
    '''

    return np.random.randint(2)

def epsilon_episode_decay(initial_epsilon, min_epsilon, rate=500):
    '''
    Linearly decays epsilon
    '''

    # Decay epsilon to some min value according to episode
    return lambda episode: max(initial_epsilon - episode / rate, min_epsilon)


def epsilon_exponential_decay(initial_epsilon, min_epsilon, rate=.995):
    '''
    Linearly decays epsilon
    '''

    # Decay epsilon to some min value according to episode
    return lambda episode: max(initial_epsilon * (rate**episode), min_epsilon)




