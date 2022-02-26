import numpy as np


class FourRooms:
    """
    A toy grid world example for TAPRL:

    Source environment (original):
        Rewards are distributed with goal state at [8, 8]

    Target environment 1 (v1):
       Rewards are distributed with goal state at [9, 9]

    Target environment 2 (v2):
       Rewards are distributed with goal state at [9, 6]
    """

    def __init__(self, env_type, mu=0.0, sigma=10.0, nrows=11, ncols=11, B=200.):
        self.nrows = nrows
        self.ncols = ncols
        self.reward_type, self.goal_type, self.subgoal_type, = env_type
        self.sigma = sigma
        self.mu = mu
        self.GOAL_STATE = None
        self.B = B
        self.subgoal_reward = +225
        self.goal_reward = +250

        # Define action space: eight actions ->
        # up, down, left, right, SE, SW, NE, NW
        self.actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
        self.action_space = len(self.actions)

    def rewards_distribution(self):
        if self.goal_type == 'original':
            self.GOAL_STATE = [8, 8]
        elif self.goal_type == 'v1':
            self.GOAL_STATE = [9, 9]
        elif self.goal_type == 'v2':
            self.GOAL_STATE = [9, 6]
        else:
            raise Exception('wrong goal state!')

        x1c, x2c = self.GOAL_STATE
        x1 = np.arange(0, self.nrows, 1)
        x2 = np.arange(0, self.ncols, 1)
        X1, X2 = np.meshgrid(x1, x2)
        Y = self.B - ((X1 - x1c) ** 2 + (X2 - x2c) ** 2)

        # add gaussian noise if the reward is stochastic
        if self.reward_type == 'target':
            for i in range(self.nrows):
                for j in range(self.ncols):
                    # Y[i, j] = Y[i, j] + np.random.normal(self.mu, self.sigma)
                    Y[i, j] = Y[i, j]

        rewards = np.copy(Y)
        # walls of rooms
        rewards[:, 5] = -50 * np.ones(Y[:, 5].shape)
        rewards[5, 0:5] = -50 * np.ones(Y[5, 0:5].shape)
        rewards[6, 5:] = -50 * np.ones(Y[6, 5:].shape)
        if self.subgoal_type == 'original':
            # options (hallway subgoals)
            rewards[5, 1] = rewards[6, 8] = rewards[2, 5] = rewards[9, 5] = self.subgoal_reward
        elif self.subgoal_type == 'v1':
            # options (hallway subgoals)
            rewards[5, 2] = rewards[6, 9] = rewards[3, 5] = rewards[8, 5] = self.subgoal_reward
        else:
            raise Exception('wrong subgoal states!')

        # modify goal state
        goal_idx = self.GOAL_STATE
        rewards[goal_idx[0], goal_idx[1]] = self.goal_reward

        return rewards, Y

    def step(self, state, action):
        new_state = [state[0], state[1]]
        new_state[0] = min(max(state[0] + action[0], 0), 10)
        new_state[1] = min(max(state[1] + action[1], 0), 10)

        rewards, _ = self.rewards_distribution()
        reward = rewards[state[0]][state[1]]

        return new_state, reward


if __name__ == "__main__":
    env_type = ['source', 'original', 'original']
    env = FourRooms(env_type=env_type)
    for i in range(10):
        s_prev = [np.random.randint(11), np.random.randint(11)]
        a = [np.random.randint(4), np.random.randint(4)]
        s, r = env.step(s_prev, a)
        print(f's_t: {s_prev}, a_t: {a}, r: {r}, s_t+1: {s}')





