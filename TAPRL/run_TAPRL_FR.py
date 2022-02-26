# import libraries
from TAPRL_algo import TAPRL
from greedy_policy import greedy_policy
from algorithms.policy import optimal_policy
from environments.four_room import FourRooms
from visualization.visualizations_FR import *
from algorithms.random_sample_Q_learning import random_sample_Q_learning
from algorithms.policy import optimal_policy

# ------------------------------------------------------------------

# # ----- define parameters -------
# x0 = [0, 0]  # initial state
# num_of_options = 5   # number of options created at each state
# epsilon = 0.75  # exploration for probabilistic policy reuse
# H = 7  # horizon length of option
# Tb = 25  # number of timesteps to run the algorithm
# N = 1  # number of experiments to run
#
# # ----- run algorithm, each output is a dictionary -----
# states_cache, trajectory_cache, rewards_cache, optimal_option_cache = TAPRL(
#     'four_room', x0=x0, H=H, num_of_options=num_of_options,
#     epsilon=epsilon, Tb=Tb, N=N)

# ----------------------------------------------------
# ------------ post processing -----------------------
# ----------------------------------------------------

# visualize reward and environment
name = 'source'
env_type = [name, 'original', 'original']
env = FourRooms(env_type=env_type)
Y, _ = env.rewards_distribution()
viz = Visualizations()
viz.four_rooms_viz(name, Y)


# # get optimal policy from the Q-table ----
# name = 'target'
# env_type = [name, 'v1', 'v1']
#
# # visualize optimal policy
# env = FourRooms(env_type=env_type)
# Y, _ = env.rewards_distribution()
#
# # plot optimal policy
# viz = Visualizations()
# viz.four_rooms_viz(Y)
# # viz.policy_primitive_viz(name, Y, trajectory_cache[0], optimal_option_cache[0])
#
