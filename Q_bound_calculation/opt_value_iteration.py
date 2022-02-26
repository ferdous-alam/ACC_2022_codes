import numpy as np
from environments.four_room import FourRooms
from visualization.visualizations_FR import Visualizations
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt


def get_optimal_option(state):
    # initialize environment
    env_type = ['source', 'original', 'original']
    env = FourRooms(env_type)
    actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]

    # load optimal value functions from value iteration
    V_opt = np.load('optimal_value_functions.npy')
    V_r = V_opt.reshape(11, 11)
    opt_option = []
    for j in range(1):
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
        opt_option.append(opt_action)
    return opt_option


# initialize environment
env_type = ['source', 'original', 'original']
env = FourRooms(env_type)

x = np.arange(0, 11, 1)
y = np.arange(0, 11, 1)
X, Y = np.meshgrid(x, y)
states = []
for i in range(len(x)):
    for j in range(len(y)):
        state = [X[i][j], Y[i][j]]
        states.append(state)

actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]

# -------- initialize value function -----
V = np.zeros(len(states))
gamma = 0.98  # discount factor

theta = 1e-1  # initialize threshold theta to random value
t = 0

threshold = 1e5
while threshold > theta:
    threshold = 0
    for i in range(len(states)):
        state = states[i]
        v = V[i]
        val_cache = []
        actions_cache = []
        option = get_optimal_option(state)
        rewards_cache = []
        opt_states = []
        gamma_r = gamma
        for k in range(len(option)):
            gamma_r = gamma_r**k
            next_state, reward = env.step(state, option[k])  # deterministic transition
            reward = gamma_r * reward
            opt_states.append(next_state)
            rewards_cache.append(reward)

        opt_rewards = sum(rewards_cache)
        j = states.index(opt_states[-1])
        V[i] = np.sum((opt_rewards + (gamma**1) * V[j]))
        threshold = max(threshold, abs(v - V[i]))
    print(f'Delta: {threshold}')
# np.save('optimal_value_functions.npy', V)

# visualize value_function
# V = np.load('optimal_value_functions.npy')
viz = Visualizations()
viz.plot_value_functions(V)



