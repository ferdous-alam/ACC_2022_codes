import numpy as np


def option_utility(options_set, options_states_set, options_rewards_set):
    utility = []
    for i in range(len(options_rewards_set)):
        u = options_rewards_set[i][-1]
        utility.append(u)
    best_option_idx = np.argmax(utility)

    return best_option_idx


def get_best_option(options_set, options_states_set, options_rewards_set):
    best_option_idx = option_utility(options_set, options_states_set, options_rewards_set)
    # get information of best option
    state = options_states_set[best_option_idx][-2]
    action = options_set[best_option_idx][-1]
    next_state = options_states_set[best_option_idx][-1]
    reward = options_rewards_set[best_option_idx][-1]

    return best_option_idx, state, action, next_state, reward
