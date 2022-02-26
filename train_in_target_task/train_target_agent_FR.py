import numpy as np
from environments.four_room import FourRooms
from visualization.visualizations_FR import *
from algorithms.random_sample_Q_learning import random_sample_Q_learning
from algorithms.policy import optimal_policy


# train agent
name = 'target'
env_type = [name, 'v1', 'v1']
Q_table, trajectory = random_sample_Q_learning('four_room', env_type,
                                               [0.5, 0.98, 0.5], 100000)
np.save('../data/Q_trained_{}_FR.npy'.format(name), Q_table)
# # get optimal states from the trained Q-table
state = [0, 0]
# load Q-table
Q_table = np.load('../data/Q_trained_{}_FR.npy'.format(name))

# visualize optimal policy
env = FourRooms(env_type=env_type)
Y, _ = env.rewards_distribution()

optimal_states, optimal_rewards, optimal_actions = optimal_policy(
    'four_room', env_type, Q_table, state, 25)
# plot optimal policy
viz = Visualizations()
viz.policy_primitive_viz(name, Y, optimal_states, optimal_actions)
