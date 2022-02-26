import numpy as np


class PnCMfg:
    def __init__(self, env_type):
        """
        PnC manufacturing environment
        """
        self.env_type = env_type

        if env_type == 'source':
            self.rewards_dist = np.load('../data/source_reward_model.npy')
        elif env_type == 'target':
            rand_num = np.random.choice(1000)   # to ensure stochastic rewards
            self.rewards_dist = np.load('../data/target_reward_model_{}.npy'.
                                        format(rand_num))
        else:
            raise Exception('Invalid environment type, please choose between '
                            'deterministic/model/stochastic')

    def find_new_state(self, state, action):
        """
        Check if the current state is at the boundary,
        if not, then choose the next state and make
        the full state, otherwise stay at the previous state
        :rtype: object
        :return: new state
        """

        next_state = [0, 0]  # initialize next state
        next_state[0] = min(max(state[0] + action[0], 0), 67)
        next_state[1] = min(max(state[1] + action[1], 0), 67)

        reward = self.rewards_dist[next_state[0], next_state[1]]

        return next_state, reward

    def reset(self):
        lxy_idx, dia_idx = np.random.choice(68), np.random.choice(68)
        curr_state = [lxy_idx, dia_idx]
        reward = self.rewards_dist[curr_state[0], curr_state[1]]
        return curr_state, reward

    def step(self, state, action):
        return self.find_new_state(state, action)


if __name__ == '__main__':
    env = PnCMfg('deterministic')
    for i in range(100):
        state, reward = env.reset()
        print('s: {}, r: {}'.format(state, reward))
