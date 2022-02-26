import numpy as np
from visualization.visualizations_PnGMfg import *
from algorithms.random_sample_Q_learning import random_sample_Q_learning
from algorithms.policy import optimal_policy


# train agent
# Q_table, trajectory = random_sample_Q_learning('autonomous_mfg', 'source',
#                                                [0.5, 0.98, 0.5], 1000000)
# np.save('../data/Q_trained_source.npy', Q_table)
# get optimal states from the trained Q-table
state = [45, 45]
# Q_table = np.load('Q_hat.npy')
Q_table = np.load('../data/Q_trained_source.npy')
optimal_states, optimal_rewards, optimal_actions = optimal_policy(
    'autonomous_mfg', 'source', Q_table, state, 100)
# plot optimal policy
plot_single_optimal_policy('source', optimal_states, fig_name=state)
# plot_value_function(Q_table)
