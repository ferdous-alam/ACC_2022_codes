import numpy as np
from environments.four_room import FourRooms
from visualization.visualizations_FR import Visualizations
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


# initialize environment
env_type = ['source', 'original', 'original']
env = FourRooms(env_type)

# x = np.arange(0, 11, 1)
# y = np.arange(0, 11, 1)
# X, Y = np.meshgrid(x, y)
# states = []
# for i in range(len(x)):
#     for j in range(len(y)):
#         state = [X[i][j], Y[i][j]]
#         states.append(state)
#
# actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
#
# # -------- initialize value function -----
# V = np.zeros(len(states))
# gamma = 0.98  # discount factor
#
# theta = 1e-5  # initialize threshold theta to random value
# t = 0
#
# Delta = 1e5
# while Delta > theta:
#     Delta = 0
#     for i in range(len(states)):
#         state = states[i]
#         v = V[i]
#         val_cache = []
#         actions_cache = []
#         for action in actions:
#             next_state, reward = env.step(state, action)  # deterministic transition
#             j = states.index(next_state)
#             val = np.sum((reward + gamma * V[j]))
#             val_cache.append(val)
#             actions_cache.append(action)
#
#         max_V_idx = val_cache.index(max(val_cache))
#         V[i] = val_cache[max_V_idx]
#         Delta = max(Delta, abs(v - V[i]))
#     print(f'Delta: {Delta}')
# np.save('optimal_value_functions.npy', V)
#
# # visualize value_function
# V = np.load('optimal_value_functions.npy')
# viz = Visualizations()
# viz.plot_value_functions(V)
#

# get optimal policy
# execute optimal policy for T timesteps as this is a continuing task
V = np.load('optimal_value_functions.npy')
V_r = V.reshape(11, 11)
T = 25   # length of optimal policy

actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
opt_policy = []
state = [0, 0]
for i in range(T):
    V_opt_cache = []
    actions_cache = []
    for j in range(len(actions)):
        action = actions[j]
        next_state, reward = env.step(state, action)
        v_val = V_r[next_state[0], next_state[1]]
        V_opt_cache.append(v_val)
        actions_cache.append(action)
    opt_val_idx = np.argmax(V_opt_cache)
    opt_action = actions_cache[opt_val_idx]
    opt_policy.append(opt_action)


