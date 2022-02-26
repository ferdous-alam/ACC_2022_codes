# import libraries
from algorithms.policy import optimal_policy
from environments.four_room import FourRooms
from visualization.visualizations_FR import *
from algorithms.random_sample_Q_learning import random_sample_Q_learning
from algorithms.policy import optimal_policy


def get_optimal_option(state, Delta, env_type):
    # initialize environment
    env = FourRooms(env_type)
    actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]

    # load optimal value functions from value iteration
    V_opt = np.load('optimal_value_functions.npy')
    V_r = V_opt.reshape(11, 11)
    opt_option = []
    for i in range(Delta):
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
        opt_next_state, _ = env.step(state, opt_action)
        # print('s_t: {}, a: {}, s_t+1: {}'.format(state, opt_action, opt_next_state))
        opt_option.append(opt_action)
        state = opt_next_state

    return opt_option


def get_option_value(env_type, gamma, Delta):
    # initialize environment
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
    V_o = np.zeros(len(states))
    all_rewards = np.zeros(len(states))

    theta = 1e-5  # initialize threshold theta to random value

    threshold = 1e5
    while threshold > theta:
        threshold = 0
        for i in range(121):
            state = states[i]
            s_0 = state
            v = V_o[i]
            # create optimal option
            mu_opt = get_optimal_option(state, Delta, env_type)
            rewards_cache = []
            option_states = []
            gamma_r = gamma
            for h in range(len(mu_opt)):
                gamma_r = gamma_r ** h
                next_state, reward = env.step(state, mu_opt[h])  # deterministic transition
                discounted_reward = gamma_r * reward
                rewards_cache.append(discounted_reward)
                option_states.append(state)
                # print(f"s_t:{state}, s_t+1:{next_state}, r:{reward}")
                state = next_state
            option_states.append(state)
            reward_val = sum(rewards_cache)
            all_rewards[i] = reward_val
            j = states.index(option_states[-1])
            V_o[i] = np.sum(reward_val + ((gamma**Delta) * V_o[j]))
            threshold = max(threshold, abs(v - V_o[i]))
            # if s_0 == [8, 8] or s_0 == [10, 10]:
            # print(f's_t:{s_0}, s_t+1: {state}, rewards: {reward_val}, V:{V_o[i]}')
        print(f'Threshold: {threshold}', end='\r')
    np.save('optimal_option_value_functions_{}_gamma_{}_Delta_{}.npy'.format(
        env_type[0], gamma, Delta), V_o)
    np.save('optimal_option_rewards_{}_gamma_{}_Delta_{}.npy'.format(
        env_type[0], gamma, Delta), V_o)

    # visualize value_function
    # V_o = np.load('optimal_option_value_functions.npy')
    # viz = Visualizations()
    # viz.plot_value_functions(V_o)

    return V_o, all_rewards





