from tqdm import tqdm
from create_options import CreateOptions
from option_utility import *
from update_Q_table import update_Q_table
from visualization.visualizations_PnGMfg import *


def TAPRL(env_name, x0, H, num_of_options, epsilon, Tb, N=None):
    """
    # params --------------------------------------
    H = 20   # horizon
    epsilon = 0.75  # exploration
    num_of_options = 5  # number of created options
    # -----------------------------------------------

    """
    all_states = []
    if env_name == 'autonomous_mfg':
        all_x0 = np.load('../data/all_x0_PnCMfg.npy')

    # if run for 1 experiment, N = 1
    if N is None:
        N = 1
    # initialize
    visited_states_cache_flatten = {}
    trajectory_flatten = {}
    optimal_rewards_cache = {}
    optimal_option_flatten = {}

    # run experiment
    for exp_num in range(N):
        Q_trained_source = np.load('../data/Q_trained_source.npy')
        Q_table = Q_trained_source

        if x0 == 'random':
            if env_name == 'autonomous_mfg':
                x0 = all_x0[exp_num]  # [np.random.randint(67), np.random.randint(67)]
                x0 = [int(x0[0]), int(x0[1])]
            elif env_name == 'four_room':
                x0 = [np.random.randint(11), np.random.randint(11)]
            else:
                raise Exception('Wrong environment name: four_room or autonomous_mfg')

        state_init = x0
        trajectory = []
        all_rewards = []
        visited_states_cache = []
        optimal_options = []
        for k in range(Tb):
            create_options = CreateOptions(
                env_name, state_init, Q_table, H, epsilon, num_of_options)
            options_set, options_rewards_set, options_states_set = \
                create_options.create_options()

            # keep track of all visited states
            visited_states = [options_states_set[i][-1] for i in
                              range(len(options_states_set))]

            Q_updated = update_Q_table(
                env_name, Q_table, options_set,
                options_states_set, options_rewards_set)

            # choose best option
            best_option_idx, state, action, next_state, reward = get_best_option(
                options_set, options_states_set, options_rewards_set)

            plot_options(options_states_set, best_option_idx)

            next_state_init = next_state
            # print for debugging
            # print(f's_t:{state_init}, s_t+1:{next_state_init}')
            # print(f'{options_states_set[best_option_idx]}')
            all_states.append(state_init)
            all_rewards.append(reward)
            state_init = next_state_init
            Q_table = Q_updated
            trajectory.append(options_states_set[best_option_idx])
            best_option = options_set[best_option_idx]
            # append visited states
            visited_states_cache.append(visited_states)
            optimal_options.append(best_option)
        all_states.append(state_init)

        visited_states_cache_flatten[exp_num] = [j for sub in visited_states_cache
                                                 for j in sub]
        optimal_option_flatten[exp_num] = [j for sub in optimal_options
                                           for j in sub]

        # find total unique states
        unique_states = []
        for s in visited_states_cache_flatten:
            if s not in unique_states:
                unique_states.append(s)
        # print(f'total visited states: {len(visited_states_cache_flatten)}, '
        #       f'visited unique states: {len(unique_states)}')

        trajectory_flatten[exp_num] = [j for sub in trajectory for j in sub]
        optimal_rewards_cache[exp_num] = all_rewards

        # update Q-table
        np.save('../data/Q_table_updated_FR.npy', Q_table)

    return visited_states_cache_flatten, \
           trajectory_flatten, optimal_rewards_cache, optimal_option_flatten
