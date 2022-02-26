import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PnCMfg_data_preprocessing import *
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PnCMfg_data_preprocessing import build_data
from matplotlib import cm
from celluloid import Camera
import time

# ----------------------------------------------------------------------------------
# Use latex font for each plot, comment this section out if latex is not supported

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


# -----------------------------------------------------------------------------------


def plot_3D_data(X1, X2, Y):
    fig = plt.figure(figsize=(12, 10))
    ax = Axes3D(fig)
    ax.scatter(X1, X2, Y, 'o', 20)
    ax.xaxis.labelpad = 30
    ax.yaxis.labelpad = 30
    ax.zaxis.labelpad = 30
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_xlabel('$l_{xy}$', fontsize=65)
    ax.set_ylabel('$d$', fontsize=65)
    ax.set_zlabel('$R$', rotation=90, fontsize=65)
    plt.savefig("../figures/autonomous_mfg_3D_model_{}.pdf".format(time.strftime("%m%d-%H%M%S")), dpi=1200,
                bbox_inches='tight')
    plt.show()


def plot_2D_data(Y, plot_num=None):
    X1, X2 = np.load('../data/X1.npy'), np.load('../data/X2.npy')
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # plot agent final positions
    cs = plt.contourf(X1, X2, Y, cmap=plt.get_cmap('viridis'))
    cs = plt.contourf(X1, X2, Y, cmap=plt.get_cmap('viridis'))
    cbar = plt.colorbar(cs)
    # cbar.ax.set_ylabel('estimated rewards', fontsize=40)
    cbar.ax.tick_params(labelsize=20)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(28)

    plt.xlabel(r'$x_1$', fontsize=65)
    plt.ylabel(r'$x_2$', fontsize=65)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rcParams['axes.linewidth'] = 2.0

    # save figure as pdf
    if plot_num is not None:
        plt.savefig("../figures/autonomous_mfg_2D_model_{}.pdf".format(
            plot_num), dpi=1200, bbox_inches='tight')
    print('done!')
    plt.show()


def animate_2D_data(X1, X2, frames):
    camera = Camera(plt.figure(figsize=(10, 8)))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # plot agent final positions
    Y = np.load('GP_reward_real_0.npy')
    cs = plt.contourf(X1, X2, Y, cmap=plt.get_cmap('viridis_r'))
    for m in range(1, frames):
        Y = np.load('/GP_reward_real_{}.npy'.format(m))
        cs = plt.contourf(X1, X2, Y, cmap=plt.get_cmap('viridis_r'))
        plt.xlabel(r'$x_1$', fontsize=65)
        plt.ylabel(r'$x_2$', fontsize=65)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
        plt.rcParams['axes.linewidth'] = 2.0
        camera.snap()
    anim = camera.animate(blit=True)
    anim.save('reward_map_animation.gif', writer='imagemagick')
    plt.show()


def plot_single_optimal_policy(env_type1, states1,
                               env_type2=None, states2=None,
                               visited_states_cache_flatten=None, fig_name=None):
    # ----------------------------------------------------
    # plot properties
    plt.figure(figsize=(12, 10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # -----------------------------------------------------

    # build model
    X1, X2 = np.load('../data/X1.npy'), np.load('../data/X2.npy')
    if env_type1 == "source":
        rewards_dist1 = np.load('../data/source_reward_model.npy')
    elif env_type1 == "target":
        rewards_dist1 = np.load('../data/target_reward_model_16.npy')
    else:
        raise Exception('Wrong environment type!')

    # plot heat map
    plt.contourf(X1, X2, rewards_dist1, cmap=plt.get_cmap('viridis'))  # viridis_r

    if env_type2 is not None:
        if env_type2 == "source":
            rewards_dist2 = np.load('../data/source_reward_model.npy')
        elif env_type2 == "target":
            rewards_dist2 = np.load('../data/target_reward_model_16.npy')
        else:
            raise Exception('Wrong environment type!')
        plt.contourf(X1, X2, rewards_dist2, cmap=plt.get_cmap('viridis'))  # viridis_r

    # ----- color bar properties: ---------
    # cbar = plt.colorbar(cs)
    # cbar.ax.set_ylabel(r'estimated rewards', fontsize=40)
    # cbar.ax.tick_params(labelsize=20)
    # for t in cbar.ax.get_yticklabels():
    #     t.set_fontsize(28)
    # --------------------------------------

    # ------------ plot optimal policy ------------------------
    x = [X1[:, 0][states1[i][0]] for i in range(len(states1))]
    y = [X2[0, :][states1[i][1]] for i in range(len(states1))]
    plt.plot(x, y, '-o', markersize=1, color='black', lw=4.0, alpha=0.5,
             label=r'$\pi^*_{TAPRL}$')
    plt.scatter(x[0], y[0], s=1000, marker='o', c='black', alpha=0.5)
    plt.scatter(x[-1], y[-1], s=1500, marker='*', c='black', alpha=1.0)

    if states2 is not None:
        x = [X1[:, 0][states2[i][0]] for i in range(len(states2))]
        y = [X2[0, :][states2[i][1]] for i in range(len(states2))]
        plt.plot(x, y, '--', color='red', lw=4.0, alpha=1.0,
                 label=r'$\pi^*_\mathcal{S}$')
        plt.scatter(x[-1], y[-1], s=1500, marker='*', c='red', alpha=1.0)

    # plot visited states if provided
    if visited_states_cache_flatten is not None:
        x = [X1[:, 0][visited_states_cache_flatten[i][0]] for i in range(
            len(visited_states_cache_flatten))]
        y = [X2[0, :][visited_states_cache_flatten[i][1]] for i in range(
            len(visited_states_cache_flatten))]
        plt.scatter(x, y, s=100, marker='o', c='black', alpha=0.25,
                    label=r'$visited \ \ states$')

    # plot initial state
    initial_state = states1[0]
    state = [X1[:, 0][initial_state[0]], X1[0, :][initial_state[1]]]
    plt.xlabel(r'$x_1$', fontsize=65)
    plt.ylabel(r'$x_2$', fontsize=65)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.rcParams['axes.linewidth'] = 2.0
    if fig_name is not None:
        plt.savefig("../figures/PnCMfg_single_optimal_policy_x0_{}_exp_{}.pdf".format(
            state, fig_name), dpi=1200, bbox_inches='tight')
    plt.show()


def plot_final_state(env_type1, states, init_state):
    # ----------------------------------------------------
    # plot properties
    plt.figure(figsize=(12, 10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # -----------------------------------------------------

    # build model
    X1, X2 = np.load('../data/X1.npy'), np.load('../data/X2.npy')
    if env_type1 == "source":
        rewards_dist1 = np.load('../data/source_reward_model.npy')
    elif env_type1 == "target":
        rewards_dist1 = np.load('../data/target_reward_model_16.npy')
    else:
        raise Exception('Wrong environment type!')

    # plot heat map
    plt.contourf(X1, X2, rewards_dist1, cmap=plt.get_cmap('viridis'))  # viridis_r

    # plot visited states if provided
    x = [X1[:, 0][states[i][0]] for i in range(len(states))]
    y = [X2[0, :][states[i][1]] for i in range(len(states))]
    plt.scatter(x, y, s=400, marker='*', c='black', alpha=0.5,
                label=r'$\mathbf{x}_T$')

    # plot initial state
    if init_state != 'random':
        plt.scatter(X1[:, 0][init_state[0]], X2[0, :][init_state[1]],
                    s=500, marker='o', c='black', alpha=0.5,
                    label=r'$\mathbf{x}_0$')

    # plot properties
    plt.xlabel(r'$x_1$', fontsize=65)
    plt.ylabel(r'$x_2$', fontsize=65)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=30)
    plt.rcParams['axes.linewidth'] = 2.0
    # save_plot
    if init_state != 'random':
        plt.savefig("../figures/opt_policy_{}_{}.pdf".format(
            init_state[0], init_state[1]), dpi=1200, bbox_inches='tight')
    else:
        plt.savefig("../figures/opt_policy_{}.pdf".format(
            init_state), dpi=1200, bbox_inches='tight')

    plt.show()


def plot_multiple_optimal_policy(all_states, rewards_dist, initial_state):
    # build model
    X1, X2, _ = build_data.build_dataset()

    # plot properties
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # plot heat map
    cs = plt.contourf(X1, X2, rewards_dist, cmap=plt.get_cmap('viridis_r'))
    cbar = plt.colorbar(cs)
    # cbar.ax.set_ylabel(r'estimated rewards', fontsize=40)
    cbar.ax.tick_params(labelsize=20)
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(28)

    # plot optimal policy
    for m in range(len(all_states)):
        states = all_states[m]

        x = [X1[:, 0][states[i][0]] for i in range(len(states))]
        y = [X2[0, :][states[i][1]] for i in range(len(states))]
        plt.plot(x, y, '-o', markersize=7.5, lw=3.0, color='white', alpha=0.5)
        # plt.scatter(x[0], y[0], s=300, marker='s', c='white', alpha=0.5, label=r'initial state')
        plt.scatter(x[-1], y[-1], s=600, marker='*', c='white', alpha=0.25, label=r'optimal state')
        # plt.legend(fontsize=18)

    state = [X1[:, 0][initial_state[0]], X1[0, :][initial_state[1]]]
    plt.xlabel(r'$x_1$', fontsize=65)
    plt.ylabel(r'$x_2$', fontsize=65)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rcParams['axes.linewidth'] = 2.0
    plt.savefig("../figures/"
                "autonomous_mfg_policy_{}_{}.pdf".format(
        time.strftime("%m%d-%H%M%S"), state), dpi=1200, bbox_inches='tight')
    plt.show()


def plot_rewards(optimal_reward_cache, N, fig_name=None):
    # plot properties
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 1.5

    if len(optimal_reward_cache) == 1:
        y = optimal_reward_cache[0]
        y_avg = np.convolve(y, np.ones((N,)) / N, mode='valid')
        x_axis = [i for i in range(len(y_avg))]
        plt.plot(x_axis, y_avg, color='gray', lw=3.0)
    else:
        averaged_reward = []
        for m in range(len(optimal_reward_cache)):
            y = optimal_reward_cache[m]
            y_avg = np.convolve(y, np.ones((N,)) / N, mode='valid')
            averaged_reward.append(y_avg)
        # get the mean and std deviation of the optimal rewards
        mean = np.mean(averaged_reward, axis=0)
        std = np.std(averaged_reward, axis=0)
        #

        #
        x_axis = [i for i in range(len(averaged_reward[0]))]
        plt.fill_between(x_axis, mean + 1 * std, mean - 1 * std,
                         color='gray', alpha=0.2)
        plt.plot(x_axis, mean, color='gray', lw=4.0)

    plt.xlabel(r'timestep', fontsize=35)
    plt.ylabel(r'reward', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    if fig_name is not None:
        plt.savefig("../figures/PnCMfg_policy_{}.pdf".format(fig_name), dpi=1200, bbox_inches='tight')
    plt.show()


def plot_compare_rewards(rewards1=None, rewards2=None,
                         rewards3=None, fig_name=None):
    """
    rewards1 : 100 x 100 dictionary ---> actual obtained rewards
    rewards2 : 1 x 100 arrays   ---> offline policy rewards
    rewards3 : 1 x 100 arrays   ---> benchmark target optimal policy rewards
    """

    X1, X2 = np.load('../data/X1.npy'), np.load('../data/X2.npy')
    # get the mean and std deviation of the optimal rewards
    # convert dictionary to list
    rewards_TAPRL = []
    rewards_benchmark = []
    rewards_offline = []

    for i in range(len(rewards1)):
        rewards_TAPRL.append(rewards1[i])
        if rewards2 is not None:
            rewards_benchmark.append(rewards2[i])
        if rewards3 is not None:
            rewards_offline.append(rewards3[i])

    # get mean and std ------------------------------------
    mean_reward_TAPRL = np.mean(rewards_TAPRL, axis=0)
    std_reward_TAPRL = np.std(rewards_TAPRL, axis=0)
    mean_reward_benchmark = np.mean(rewards_benchmark, axis=0)
    std_reward_benchmark = np.std(rewards_benchmark, axis=0)
    mean_reward_offline = np.mean(rewards_offline, axis=0)
    std_reward_offline = np.std(rewards_offline, axis=0)
    # -----------------------------------------------------

    # plot properties
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 2.0

    x_axis = [i for i in range(len(rewards1[0]))]
    # plot simulated rewards: does not need error bar
    plt.plot(x_axis, mean_reward_TAPRL, color='red', lw=3.0, label=r'$E[R]$ (TAPRL)')
    # show results for 1 standard deviation limit
    plt.fill_between(x_axis, mean_reward_TAPRL + 1 * std_reward_TAPRL,
                     mean_reward_TAPRL - 1 * std_reward_TAPRL,
                     color='blue', alpha=0.1)

    # plot experimental rewards error bar
    if rewards2 is not None:  # target optimal
        plt.plot(x_axis, mean_reward_benchmark, color='blue', lw=3.0, label=r'$E[R]$ (benchmark)')
        plt.fill_between(x_axis, mean_reward_benchmark + 1 * std_reward_benchmark,
                         mean_reward_benchmark - 1 * std_reward_benchmark,
                         color='blue', alpha=0.1)
    if rewards3 is not None:  # source optimal
        plt.plot(x_axis, mean_reward_offline, color='black', lw=3.0, label=r'$E[R]$ (offline)')
        plt.fill_between(x_axis, mean_reward_offline + 1 * std_reward_offline,
                         mean_reward_offline - 1 * std_reward_offline,
                         color='blue', alpha=0.1)

    # x_axis_label = np.arange(0, len(mean_reward_TAPRL), 5)
    plt.xticks()
    plt.legend(fontsize=25)
    plt.xlabel(r'timestep', fontsize=35)
    plt.ylabel(r'reward', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.savefig("../figures/opt_policy_rewards_{}_{}.pdf".format(
        fig_name[0], fig_name[1]),
        dpi=1200, bbox_inches='tight')
    plt.show()


def plot_multiple_init_rewards(rewards):
    # plot properties
    plt.figure(figsize=(12, 8))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 2.0

    n_vals = [5, 10, 15]
    for t in range(len(rewards)):
        mean_real = np.mean(rewards[t], axis=0)
        std_real = np.std(rewards[t], axis=0)
        # plot simulated rewards: does not need error bar
        x_axis = [i for i in range(len(rewards[t][0]))]
        plt.plot(x_axis, mean_real,
                 lw=3.0, label='H={}'.format(n_vals[t]))

        plt.fill_between(x_axis, mean_real + 1 * std_real,
                         mean_real - 1 * std_real, alpha=0.1)

    plt.xlabel(r'timestep', fontsize=35)
    plt.ylabel(r'reward', fontsize=35)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.legend(fontsize=25)
    plt.ylim([0, 80])
    # ---------- save plot --------
    plt.savefig("../figures/multi_init_rewards_{}.pdf".format(
        time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')
    # -----------------------------
    plt.show()


def plot_options(options_states_set, best_option_idx=None):
    # ----------------------------------------------------
    # plot properties
    plt.figure(figsize=(12, 10))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    # -----------------------------------------------------

    X1, X2 = np.load('../data/X1.npy'), np.load('../data/X2.npy')
    Y = np.load('../data/target_reward_model_16.npy')

    # plot agent final positions
    cs = plt.contourf(X1, X2, Y, cmap=plt.get_cmap('viridis'))
    cs = plt.contourf(X1, X2, Y, cmap=plt.get_cmap('viridis'))

    # ------------ plot optimal policy ------------------------
    x, y = [], []
    for k in range(len(options_states_set)):
        x = [X1[:, 0][options_states_set[k][i][0]] for i in range(
            len(options_states_set[k]))]
        y = [X2[0, :][options_states_set[k][i][1]] for i in range(
            len(options_states_set[k]))]
        plt.plot(x, y, '-o', markersize=1, lw=4.0, color='black', alpha=0.75)
        plt.scatter(x[0], y[0], s=600, marker='o', c='black', alpha=0.75)
        plt.scatter(x[-1], y[-1], s=800, marker='*', color='black', alpha=0.75)
    # legend purpose
    plt.plot(x, y, '-o', markersize=1, lw=4.0, color='black', label=r'$o_i$')
    plt.scatter(x[0], y[0], s=600, marker='o', c='black',
                alpha=0.75, label=r'$\mathbf{x}_0$')
    plt.scatter(x[-1], y[-1], s=800, marker='*',
                alpha=0.75, color='black', label=r'$subgoal$')

    if best_option_idx is not None:
        x = [X1[:, 0][options_states_set[best_option_idx][i][0]] for i in range(
            len(options_states_set[best_option_idx]))]
        y = [X2[0, :][options_states_set[best_option_idx][i][1]] for i in range(
            len(options_states_set[best_option_idx]))]
        plt.plot(x, y, '-o', markersize=1, color='red', lw=4.0,
                 alpha=0.5, label=r'$o_b$')

    plt.xlabel(r'$x_1$', fontsize=65)
    plt.ylabel(r'$x_2$', fontsize=65)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.rcParams['axes.linewidth'] = 2.0
    plt.legend(fontsize=25)
    state = [X1[:, 0][options_states_set[0][0][0]], X1[0, :][options_states_set[0][0][1]]]
    plt.savefig("../figures/options_viz_{}.pdf".format(
        state), dpi=1200, bbox_inches='tight')

    plt.show()


def plot_value_function(Q_table):
    V_func = np.zeros((len(Q_table), len(Q_table)))

    for i in range(len(Q_table)):
        for j in range(len(Q_table)):
            V_func[i, j] = np.max(Q_table[i, j, :])

    # V_func = V_func / np.max(V_func)
    x, y = [i for i in range(len(Q_table))], [i for i in range(len(Q_table))]
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(10, 8))
    # ax = fig.gca(projection='3d')
    # cmap = plt.cm.get_cmap('Spectral')
    #
    # surf = ax.plot_surface(X, Y, V_func, cmap=cm.coolwarm,
    #                        linewidth=0, antialiased=False)
    # ax.zaxis.set_tick_params(labelsize=20)
    # ax.xaxis.set_tick_params(labelsize=20)
    # ax.yaxis.set_tick_params(labelsize=20)
    # plt.savefig("../figures/value_functions_{}.pdf".format(
    #     time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')

    X1, X2 = np.load('../data/X1.npy'), np.load('../data/X2.npy')
    plt.contourf(X1, X2, V_func, cmap=plt.get_cmap('viridis'))  # viridis_r
    plt.savefig("../figures/value_functions_{}.pdf".format(
        time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')

    plt.show()


def plot_rewards_error_bar(variable, mean_vals, std_vals):
    """
    variable: i.e. H = [1, 2, 3, 4, 5]
    mean_vals: i.e. mean_vals = [mean_reward_TAPRL, mean_reward_benchmark,
                                                    mean_reward_offline]

    std_vals: i.e. std_vals = [std_reward_TAPRL, std_reward_benchmark,
                                                    std_reward_offline]
    """

    TAPRL = [mean_vals[i][0] for i in range(len(mean_vals))]
    benchmark = [mean_vals[i][1] for i in range(len(mean_vals))]
    offline = [mean_vals[i][2] for i in range(len(mean_vals))]
    std1 = [std_vals[i][0] for i in range(len(std_vals))]
    std2 = [std_vals[i][1] for i in range(len(std_vals))]
    std3 = [std_vals[i][2] for i in range(len(std_vals))]

    # plot errorbar ----------------------------
    # ----------------------------------------------------
    # plot properties
    plt.figure(figsize=(12, 6))
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams['axes.linewidth'] = 2.0
    fig, ax = plt.subplots()
    # -----------------------------------------------------
    X = np.arange(len(variable))
    x_pos = [(X[i] + 0.25) for i in range(len(X))]
    xvalues = [r'$\epsilon$: {}'.format(variable[i]) for i in range(len(variable))]

    plt.bar(X, TAPRL, width=0.25, label=r'$\pi^*_{TAPRL}$')
    plt.bar(X + 0.25, benchmark, width=0.25, label=r'$\pi^*_\mathcal{T}$')
    plt.bar(X + 0.50, offline, width=0.25, label=r'$\pi^*_\mathcal{S}$')

    plt.errorbar(X, TAPRL, yerr=std1, fmt='.', color='black')
    plt.errorbar(X + 0.25, benchmark, yerr=std2, fmt='.', color='black')
    plt.errorbar(X + 0.50, offline, yerr=std3, fmt='.', color='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(xvalues, fontsize=15)
    plt.ylabel(r'final state reward', fontsize=18)
    plt.legend(fontsize=12)

    plt.savefig("../figures/effect_of_epsilon_value.pdf",
                dpi=1200, bbox_inches='tight')

    plt.show()


if __name__ == '__main__':
    # X1, X2, _ = build_data.build_dataset()
    # Y = np.load('/home/ghost-083/Research/Codes/CDC 2021/autonomous_mfg/'
    #             'training_data/real_model.npy')
    # plot_2D_data(X1, X2, Y)
    # # animate_2D_data(X1, X2, frames=25)
    a = np.load('horizon_5.npy')
    b = np.load('horizon_10.npy')
    c = np.load('horizon_15.npy')

    rewards = {}
    rewards[0], rewards[1], rewards[2] = a, b, c
    plot_multiple_init_rewards(rewards)
