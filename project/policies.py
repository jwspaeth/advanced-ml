
import numpy as np

def epsilon_greedy_policy(state, model, epsilon = 0):
    '''
    Simple policy which either gets the next action or random action with chance epsilon in cartpole
    '''

    n_actions = len(model.outputs)

    if np.random.rand() < epsilon:
        return np.random.randint(2, size=n_actions)
    else:
        Q_values = model.predict(state[np.newaxis])
        return [np.argmax(Q_values_one_action[0]) for Q_values_one_action in Q_values]

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