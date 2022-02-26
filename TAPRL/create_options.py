import numpy as np
from environments.PnCMfg import PnCMfg
from environments.four_room import FourRooms

from visualization.visualizations_PnGMfg import *


class CreateOptions:
    def __init__(self, env_name, state_init, Q_table, H, epsilon, num_of_options):
        self.env_name = env_name
        self.state_init = state_init
        self.Q_table = Q_table
        self.H = H   # horizon length
        self.epsilon = epsilon
        self.num_of_options = num_of_options
        # action space
        if self.env_name == 'four_room':
            self.actions = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]
            self.env_t = FourRooms(['target', 'v1', 'v1'])
            self.env_s = FourRooms(['source', 'original', 'original'])
        elif self.env_name == 'autonomous_mfg':
            self.actions = [[0, 1], [0, -1], [-1, 0], [1, 0], [1, 1],
                            [-1, 1], [-1, -1], [1, -1], [0, 0]]
            self.env_t = PnCMfg('target')
            self.env_s = PnCMfg('source')
        else:
            raise Exception('Unknown environment name: choose '
                            'four_room or autonomous_mfg')

    def create_option(self, Delta):

        # create options
        option = []
        option_states = []
        option_rewards = []
        state = self.state_init

        for i in range(Delta):
            rand_num = np.random.rand()
            if rand_num <= self.epsilon:
                if self.env_name == 'four_room':
                    action_idx = np.random.choice(5)
                elif self.env_name == 'autonomous_mfg':
                    action_idx = np.random.choice(9)
                else:
                    raise Exception('Wrong environment name: four_room or autonomous_mfg')
            else:
                action_idx = np.argmax(self.Q_table[state[0], state[1], :])

            action = self.actions[action_idx]
            if i == Delta-1:
                next_state, reward = self.env_t.step(state, action)
            else:
                next_state, reward = self.env_s.step(state, action)

            # append to the option
            option.append(action)
            option_states.append(state)
            option_rewards.append(reward)
            # update state
            state = next_state
        option_states.append(state)

        return option, option_rewards, option_states

    def greedy_option(self, Delta):
        # create options
        option = []
        option_states = []
        option_rewards = []
        state = self.state_init

        for i in range(Delta):
            action_idx = np.argmax(self.Q_table[state[0], state[1], :])
            action = self.actions[action_idx]
            if i == Delta-1:
                next_state, reward = self.env_t.step(state, action)
            else:
                next_state, reward = self.env_s.step(state, action)

            # append to the option
            option.append(action)
            option_states.append(state)
            option_rewards.append(reward)
            # update state
            state = next_state
        option_states.append(state)

        return option, option_states, option_rewards

    def get_subgoal(self):
        option, option_states, option_rewards = self.greedy_option(self.H)
        max_reward_idx = option_rewards.index(max(option_rewards))
        subgoal = option_states[max_reward_idx]
        Delta = max_reward_idx + 1

        return subgoal, Delta

    def create_options(self):
        # identify subgoal state
        subgoal, Delta = self.get_subgoal()
        # initialization
        options_set = {}
        options_rewards_set = {}
        options_states_set = {}

        # get greedy option and append as the first element of the options set
        option_g, option_states_g, option_rewards_g = self.greedy_option(Delta)
        options_set[0] = option_g
        options_rewards_set[0] = option_rewards_g
        options_states_set[0] = option_states_g

        for i in range(self.num_of_options - 1):
            option, option_rewards,  option_states = self.create_option(Delta)
            options_set[i+1] = option
            options_rewards_set[i+1] = option_rewards
            options_states_set[i+1] = option_states

        return options_set, options_rewards_set, options_states_set


if __name__ == "__main__":
    Q_table = np.load('../data/Q_trained_source_FR.npy')
    create_options = CreateOptions('four_room', [0, 0], Q_table, 5, 0.5, 5)
    options_set, options_rewards_set, options_states_set = create_options.create_options()

    # # viz
    # plot_options(options_states_set)




