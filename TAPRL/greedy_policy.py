import numpy as np
from environments.four_room import FourRooms
from environments.PnCMfg import PnCMfg


def greedy_policy(env_name, Q_table, opt_policy_length, state):
    # determine environment
    if env_name == 'four_room':
        env = FourRooms('source')
        # primitive actions
        actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
    elif env_name == 'autonomous_mfg':
        env = PnCMfg('source')
        # primitive actions
        actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                   [-1, 1], [-1, -1], [1, -1], [0, 0]]
    else:
        raise Exception('Environment name not defined: please choose between '
                        'four_room or autonomous_mfg')
    optimal_states = []
    for i in range(opt_policy_length):
        action_idx = np.argmax(Q_table[state[0], state[1], :])
        action = actions[action_idx]
        next_state, _ = env.step(state, action)
        optimal_states.append(state)
        # print(f's_t:{state}, a:{action}, s_t+1:{next_state}')
        state = next_state
    optimal_states.append(state)

    return optimal_states
