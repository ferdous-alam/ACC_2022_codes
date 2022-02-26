 import numpy as np


def update_Q_table(env_name, Q_table, options_set, options_states_set, options_rewards_set):
    # hyper params ---------------
    alpha = 0.90
    gamma = 0.25
    # ----------------------------
    if env_name == 'four_room':
        actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
    elif env_name == 'autonomous_mfg':
        actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                   [-1, 1], [-1, -1], [1, -1], [0, 0]]
    else:
        raise Exception('Wrong environment name: four_room or autonomous_mfg')

    option_final_states = []
    option_final_rewards = []
    for i in range(len(options_set)):
        if len(options_set[i]) >= 2:
            # extract information from created options hash table
            state = options_states_set[i][-2]
            action = options_set[i][-1]
            next_state = options_states_set[i][-1]
            reward = options_rewards_set[i][-1]
            # print(f"s_t:{state}, a_t:{action}, r:{reward}, s_t+1:{next_state}")
            action_idx = actions.index(action)
            # update Q-table
            next_best_action_idx = np.argmax(Q_table[next_state[0], next_state[1], :])
            Q_table[state[0], state[1], action_idx] += \
                alpha * (reward + gamma * Q_table[next_state[0], next_state[1], next_best_action_idx]
                         - Q_table[state[0], state[1], action_idx])
            option_final_states.append(next_state)
            option_final_rewards.append(reward)

    return Q_table
