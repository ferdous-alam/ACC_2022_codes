from tqdm import tqdm
import numpy as np
from algorithms.policy import *
from environments.PnCMfg import PnCMfg
from environments.four_room import FourRooms
from visualization.visualizations_PnGMfg import *


def random_sample_Q_learning(env_name, env_type,
                             algo_params, timesteps):
    # unpack details
    alpha, gamma, epsilon = algo_params

    # determine environment
    if env_name == 'four_room':
        x1_len, x2_len = 11, 11
        # primitive actions
        actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
    elif env_name == 'autonomous_mfg':
        x1_len, x2_len = 68, 68
        # primitive actions
        actions = [[0, +1], [0, -1], [-1, 0], [1, 0], [+1, +1],
                   [-1, 1], [-1, -1], [1, -1], [0, 0]]
    else:
        raise Exception('Environment name not defined: please choose between '
                        'four_room or autonomous_mfg')
    # --------- initialization ------------------
    trajectory = {}
    Q_table = np.zeros((x1_len, x2_len, len(actions)))
    # Q_table = np.load('../data/Q_trained_source.npy')

    # initial state
    for t in tqdm(range(timesteps)):
        # determine stochastic environment for every timestep
        if env_name == 'four_room':
            env = FourRooms(env_type)
        elif env_name == 'autonomous_mfg':
            env = PnCMfg(env_type)
        else:
            raise Exception('Environment name not defined: please choose between '
                            'four_room or autonomous_mfg')
        # ----------------------------------------------------------------

        state = [np.random.choice(x1_len), np.random.choice(x2_len)]
        action, action_idx = policy(env_name, state, Q_table, 'epsilon_greedy', epsilon)
        next_state, reward = env.step(state, action)
        next_best_action_idx = np.argmax(Q_table[next_state[0], next_state[1], :])
        Q_table[state[0], state[1], action_idx] += \
            alpha * (reward + gamma * Q_table[next_state[0], next_state[1], next_best_action_idx]
                     - Q_table[state[0], state[1], action_idx])
        trajectory[t] = {'state': state,
                         'action': action,
                         'reward': reward,
                         'next_state': next_state}

    return Q_table, trajectory


if __name__ == '__main__':
    # train agent
    Q_table, trajectory = random_sample_Q_learning('autonomous_mfg', 'source',
                                                   [0.5, 0.98, 0.5], 10000000)
    np.save('../data/Q_trained_source.npy', Q_table)
    # get optimal states from the trained Q-table
    state = [5, 45]
    Q_table = np.load('../data/Q_trained_source.npy')
    optimal_states, optimal_rewards, optimal_actions = optimal_policy(
        'autonomous_mfg', 'source', Q_table, state, 100)
    # plot optimal policy
    plot_single_optimal_policy(state, 'source', optimal_states)
    # plot_value_function(Q_table)
