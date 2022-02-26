import numpy as np
from visualization.visualizations_PnGMfg import *
from algorithms.random_sample_Q_learning import random_sample_Q_learning
from algorithms.policy import optimal_policy


# train agent
# Q_table, trajectory = random_sample_Q_learning('autonomous_mfg', 'target',
#                                                [0.5, 0.98, 0.5], 10000000)
# np.save('../data/Q_trained_target.npy', Q_table)
# get optimal states from the trained Q-table
state = [5, 5]
Q_table = np.load('../data/Q_trained_target.npy')
optimal_states, optimal_rewards, optimal_actions = optimal_policy(
    'autonomous_mfg', 'target', Q_table, state, 100)
# plot optimal policy
plot_single_optimal_policy('target', optimal_states, fig_name=state)
# plot_value_function(Q_table)
