import numpy as np
from environments.four_room import FourRooms
from environments.PnCMfg import PnCMfg


def policy(env_name, state, Q_table, policy_type, epsilon=None):
    if env_name == 'four_room':
        actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
    elif env_name == 'autonomous_mfg':
        actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                   [-1, 1], [-1, -1], [1, -1], [0, 0]]
    else:
        raise Exception('Environment name not defined: please choose between '
                        'four_room or autonomous_mfg')

    number = np.random.random_sample()
    if policy_type == 'greedy':
        action_idx = np.argmax(Q_table[state[0], state[1], :])
    elif policy_type == 'epsilon_greedy':
        if number <= epsilon:
            action_idx = np.random.choice(len(actions))
        else:
            action_idx = np.argmax(Q_table[state[0], state[1], :])
    else:
        raise Exception('Invalid policy, please choose: greedy or epsilon-greedy')
    action = actions[action_idx]

    return action, action_idx


def optimal_policy(env_name, env_type, Q_table, state, opt_policy_length):
    # --------- optimal policy ----------
    optimal_policy_info = {}

    # determine environment
    if env_name == 'four_room':
        env = FourRooms(env_type)
        # primitive actions
        actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
    elif env_name == 'autonomous_mfg':
        env = PnCMfg(env_type)
        # primitive actions
        actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                   [-1, 1], [-1, -1], [1, -1], [0, 0]]
    else:
        raise Exception('Environment name not defined: please choose between '
                        'four_room or autonomous_mfg')

    # initialization
    optimal_rewards = []
    optimal_states = []
    optimal_actions = []

    for i in range(opt_policy_length):
        action_idx = np.argmax(Q_table[state[0], state[1], 0:len(actions)])
        action = actions[action_idx]
        next_state, reward = env.step(state, action)
        optimal_rewards.append(reward)
        optimal_states.append(state)
        # print(f's_t:{state}, a:{action}, r:{reward}, s_t+1:{next_state}')
        state = next_state
        optimal_actions.append(action)
    optimal_states.append(state)

    return optimal_states, optimal_rewards, optimal_actions


if __name__ == "__main__":
    Q_table = np.load('../data/Q_trained_source.npy')
    optimal_states, optimal_rewards, optimal_actions = optimal_policy(
        'autonomous_mfg', 'source', Q_table, [65, 5], 25)



