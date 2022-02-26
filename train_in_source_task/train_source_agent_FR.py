import numpy as np
from environments.four_room import FourRooms
from visualization.visualizations_FR import *
from algorithms.random_sample_Q_learning import random_sample_Q_learning
from algorithms.policy import optimal_policy


# train agent
env_type = ['source', 'original', 'original']
Q_table, trajectory = random_sample_Q_learning('four_room', env_type,
                                               [0.5, 0.98, 0.5], 100000)
np.save('../data/Q_trained_source_FR.npy', Q_table)
# # get optimal states from the trained Q-table
state = [0, 0]
# load Q-table
Q_table = np.load('../data/Q_trained_source_FR.npy')

# visualize optimal policy
env = FourRooms(env_type=env_type)
Y, _ = env.rewards_distribution()

optimal_states, optimal_rewards, optimal_actions = optimal_policy(
    'four_room', env_type, Q_table, state, 25)
# plot optimal policy
viz = Visualizations()
viz.policy_primitive_viz('target', Y, optimal_states, optimal_actions)
