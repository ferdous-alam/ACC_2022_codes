# import libraries
from TAPRL_algo import TAPRL
from greedy_policy import greedy_policy
from tqdm import tqdm
from algorithms.policy import optimal_policy
from visualization.visualizations_PnGMfg import *
# ------------------------------------------------------------------


def get_statistics(rewards_cache_TAPRL,
                   rewards_cache_benchmark,
                   rewards_cache_offline):
    rewards_TAPRL = []
    rewards_benchmark = []
    rewards_offline = []
    for i in range(len(rewards_cache_TAPRL)):
        rewards_TAPRL.append(rewards_cache_TAPRL[i])
        rewards_benchmark.append(rewards_cache_benchmark[i])
        rewards_offline.append(rewards_cache_offline[i])

    mean_TAPRL, std_TAPRL = np.mean(rewards_TAPRL, axis=0), np.std(
        rewards_TAPRL, axis=0)
    mean_benchmark, std_benchmark = np.mean(rewards_benchmark, axis=0), np.std(
        rewards_benchmark, axis=0)
    mean_offline, std_offline = np.mean(rewards_offline, axis=0), np.std(
        rewards_offline, axis=0)

    m1, s1 = mean_TAPRL[-1], std_TAPRL[-1]
    m2, s2 = mean_benchmark[-1], std_benchmark[-1]
    m3, s3 = mean_offline[-1], std_offline[-1]

    return m1, s1, m2, s2, m3, s3


def run_experiments(x0, num_of_options=5, epsilon=0.75, H=5, Tb=100, N=100):
    """
    # # ----- define parameters -------
    x0 = 'random'   # [45, 45]   # initial state
    num_of_options = 5   # number of options created at each state
    epsilon = 0.75  # exploration for probabilistic policy reuse
    H = 5  # horizon length of option
    Tb = 100  # number of timesteps to run the algorithm
    N = 100   # number of experiments to run
    """
    # ----- run algorithm, each output is a dictionary -----
    states_cache, trajectory_cache, rewards_cache, option_cache = TAPRL(
        env_name='autonomous_mfg', x0=x0, H=H,
        num_of_options=num_of_options, epsilon=epsilon, Tb=Tb, N=N)

    # save data
    # np.save('../data/states_cache_x0_{}.npy'.format(x0), states_cache)
    # np.save('../data/trajectory_cache_x0_{}.npy'.format(x0), trajectory_cache)
    # np.save('../data/rewards_cache_x0_{}.npy'.format(x0), rewards_cache)

    # ----------------------------------------------------
    # ------------ post processing -----------------------
    # ----------------------------------------------------

    # 1) visualize reward functions ----
    # source_reward = np.load('../data/source_reward_model.npy')
    # target_reward = np.load('../data/target_reward_model_16.npy')
    # plot_2D_data(source_reward, 'source')
    # plot_2D_data(target_reward, 'target')

    # 2) get optimal policy from the Q-table ----
    Q_table = np.load('../data/Q_trained_source.npy')
    # optimal_states = greedy_policy('autonomous_mfg', Q_table, Tb, x0)
    # plot_single_optimal_policy('target', optimal_states,
    #                            fig_name=x0)

    # # 3) plot final states from all experiments ----
    final_states = [trajectory_cache[j][-1] for j in range(len(trajectory_cache))]
    # plot_final_state('target', states=final_states, init_state=x0)

    # 4) plot optimal rewards from all experiments ----
    rewards_cache_benchmark = {}
    rewards_cache_offline = {}
    opt_states_target = {}
    opt_states_source = {}

    # initial state reproducibility
    if x0 == 'random':
        # load random x0, for reproducibility a separate files have been made apriori
        all_x0 = np.load('../data/all_x0_PnCMfg.npy')
    elif x0 == [5, 5] or x0 == [45, 45]:
        all_x0 = np.load('../data/all_x0_{}_{}.npy'.format(x0[0], x0[1]))
    else:
        raise Exception('Wrong initial state: choose between [5, 5] and [45, 45]')

    for i in tqdm(range(N)):
        x0 = all_x0[i]  # [np.random.randint(67), np.random.randint(67)]
        x0 = [int(x0[0]), int(x0[1])]
        print(x0)
        Q_target = np.load('../data/Q_trained_target.npy')   # source
        opt_states_target[i], rewards_cache_benchmark[i], _ = optimal_policy('autonomous_mfg', 'target', Q_target, x0, Tb)

        Q_source = np.load('../data/Q_trained_source.npy')     # target
        opt_states_source[i], rewards_cache_offline[i], _ = optimal_policy(
            'autonomous_mfg', 'target', Q_source, x0, Tb)

    # plot_compare_rewards(rewards_cache, rewards_cache_benchmark,
    #                      rewards_cache_offline, fig_name=x0)

    # # # 5) plot trajectory
    # plot_single_optimal_policy(env_type1='target',
    #                            states1=trajectory_cache[0],
    #                            env_type2='target',
    #                            states2=opt_states_source[0],
    #                            visited_states_cache_flatten=states_cache[0],
    #                            fig_name=x0)

    return rewards_cache, rewards_cache_benchmark, rewards_cache_offline


if __name__ == "__main__":
    # variable = [0.05, 0.10, 0.25, 0.50, 0.75, 0.99]
    gamma = 0.75
    mean_vals = []
    std_vals = []
    for i in range(1):
        rewards_cache_TAPRL, rewards_cache_benchmark, rewards_cache_offline = run_experiments(
            x0=[45, 45], num_of_options=5,
            epsilon=gamma, H=10, Tb=1, N=1)

        m1, s1, m2, s2, m3, s3 = get_statistics(rewards_cache_TAPRL,
                                                rewards_cache_benchmark,
                                                rewards_cache_offline)
        mean_vals.append([m1, m2, m3])
        std_vals.append([s1, s2, s3])

    plot_compare_rewards(rewards_cache_TAPRL, rewards_cache_benchmark,
                         rewards_cache_offline, fig_name='random')

    # plot_rewards_error_bar(variable, mean_vals, std_vals)


