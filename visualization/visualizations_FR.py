import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
from environments.four_room import FourRooms
from celluloid import Camera
import time


class Visualizations:
    def __init__(self):
        x1 = np.arange(0, 11, 1)
        x2 = np.arange(0, 11, 1)
        self.X1, self.X2 = np.meshgrid(x1, x2)
        # action_list = [right, left, down, up, no_change]
        self.action_list = [[-1, 0], [+1, 0], [0, -1], [0, +1], [0, 0]]

        # ----------------------------------------------------------------------------------
        # Use latex font for each plot, comment this section out if latex is not supported
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        # -----------------------------------------------------------------------------------

    def plot_reward_distribution(self, Y, plot_type=None):
        fig = plt.figure(figsize=(12, 10))
        ax = fig.gca(projection='3d')

        # Plot the surface.
        surf = ax.plot_surface(self.X1, self.X2, Y, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # plt.savefig("figures/reward_mean.pdf", dpi=1200, bbox_inches='tight')

        # ---------------- plot 3D reward bar --------------
        if plot_type == 'bar':
            rewards = Y.ravel()
            _X = self.X1.ravel()
            _Y = self.X2.ravel()
            top = rewards
            bottom = np.zeros_like(top)
            width = depth = 1
            cmap = cm.coolwarm
            rgba = [cmap((k - np.min(top)) / np.max(top)) for k in top]
            ax.bar3d(_X, _Y, bottom, width, depth, top, color=rgba, shade=True)
            # ax.view_init(25, -145)
            # plt.draw()
        plt.show()

    def four_rooms_viz(self, env_name, Y):
        # Make a Y.shape grid...
        nrows, ncols = Y.shape
        # Set every cell to reward value
        image = Y
        # Reshape things into a (nrows x ncols) grid.
        image = image.reshape((nrows, ncols))

        fig = plt.figure(figsize=(10, 8))
        row_labels = range(nrows)
        col_labels = range(ncols)
        cax = plt.matshow(image, fignum=1)
        plt.xticks(range(ncols), col_labels)
        plt.yticks(range(nrows), row_labels)
        cbar = fig.colorbar(cax)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(20)
        plt.savefig("../figures/four_room_{}.pdf".format(env_name),
                    dpi=1200, bbox_inches='tight')
        plt.show()

    def policy_primitive_viz(self, env_name, Y, states, actions):
        # Make a Y.shape grid...
        nrows, ncols = Y.shape
        # Set every cell to reward value
        image = Y
        # Reshape things into a (nrows x ncols) grid.
        image = image.reshape((nrows, ncols))

        fig = plt.figure(figsize=(10, 8))
        row_labels = range(nrows)
        col_labels = range(ncols)
        cax = plt.matshow(image, fignum=1)
        plt.xticks(range(ncols), col_labels)
        plt.yticks(range(nrows), row_labels)
        # cbar = fig.colorbar(cax)
        # cbar.ax.tick_params(labelsize=20)
        # for t in cbar.ax.get_yticklabels():
        #     t.set_fontsize(28)

        # plot policy
        x_pos = np.zeros(len(actions))
        y_pos = np.zeros(len(actions))
        for i in range(len(actions)):
            x_pos[i] = states[i][1]
            y_pos[i] = states[i][0]
            if actions[i] == self.action_list[0]:  # UP
                x, y = x_pos[i], y_pos[i]+0.4
                dx, dy = 0, -0.8
            elif actions[i] == self.action_list[1]:  # DOWN
                x, y = x_pos[i], y_pos[i] - 0.4
                dx, dy = 0, 0.8
            elif actions[i] == self.action_list[2]:  # LEFT
                x, y = x_pos[i] + 0.4, y_pos[i]
                dx, dy = -0.8, 0
            elif actions[i] == self.action_list[3]:  # RIGHT
                x, y = x_pos[i]-0.4, y_pos[i]
                dx, dy = 0.8, 0
            elif actions[i] == self.action_list[4]:
                x, y = x_pos[i], y_pos[i]
                dx, dy = 0, 0
            else:
                raise Exception('Invalid action!')

            plt.arrow(x, y, dx, dy, length_includes_head=True,
                      head_width=0.15, head_length=0.3, lw=2.0, color='black')
        plt.scatter(states[-1][0], states[-1][1], s=300,
                    marker='o', color='black', alpha=0.5)

        plt.savefig("../figures/trained_policy_{}_FR.pdf".format(env_name),
                    dpi=1200, bbox_inches='tight')
        plt.show()

    def reward_plot(self, rewards_cache, N):
        y = rewards_cache
        # use running average (convolution filter)
        y = np.convolve(rewards_cache, np.ones((N,)) / N, mode='valid')
        x = np.arange(0, len(y), 1)

        fig = plt.figure(figsize=(12, 8))
        plt.plot(x, y, lw=2.0)
        plt.xlabel(r'timesteps', fontsize=40)
        plt.ylabel(r'reward', fontsize=40)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)

        plt.savefig("figures/reward_per_time_{}.pdf".format(time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')
        plt.show()

    def policy_options_viz(self, opt_states, env_type, goal_type,
                           subgoal_type, T=None, adjacent_states=None,
                           all_states=None, all_actions=None, opt_states_sim=None):

        env = FourRooms(env_type, goal_type, subgoal_type)
        Y, _ = env.rewards_distribution()
        # Make a Y.shape grid...
        nrows, ncols = Y.shape
        # Set every cell to reward value
        image = Y
        # Reshape things into a (nrows x ncols) grid.
        image = image.reshape((nrows, ncols))

        fig = plt.figure(figsize=(10, 8))
        row_labels = range(nrows)
        col_labels = range(ncols)
        cax = plt.matshow(image, fignum=1)
        plt.xticks(range(ncols), col_labels)
        plt.yticks(range(nrows), row_labels)
        # cbar = fig.colorbar(cax)
        # for t in cbar.ax.get_yticklabels():
        #     t.set_fontsize(20)

        # plot primitive policies
        # --------------------------------------------
        if (all_states is None) and (all_actions is None):
            flag = True
        else:
            flag = False

        if flag is False:
            for t in range(len(all_states)):
                x_pos = np.zeros(len(all_actions[t]))
                y_pos = np.zeros(len(all_actions[t]))
                for i in range(len(all_actions[t])):
                    x_pos[i] = all_states[t][i][1]
                    y_pos[i] = all_states[t][i][0]
                    if all_actions[t][i] == 0:  # UP
                        x, y = x_pos[i], y_pos[i] + 0.4
                        dx, dy = 0, -0.8
                    elif all_actions[t][i] == 1:  # DOWN
                        x, y = x_pos[i], y_pos[i] - 0.4
                        dx, dy = 0, 0.8
                    elif all_actions[t][i] == 2:  # LEFT
                        x, y = x_pos[i] + 0.4, y_pos[i]
                        dx, dy = -0.8, 0
                    elif all_actions[t][i] == 3:  # RIGHT
                        x, y = x_pos[i] - 0.4, y_pos[i]
                        dx, dy = 0.8, 0
                    elif all_actions[t][i] == 4:
                        x, y = x_pos[i], y_pos[i]
                        dx, dy = 0, 0
                    else:
                        raise Exception('Invalid action!')

                    plt.arrow(x, y, dx, dy, length_includes_head=True,
                              head_width=0.15, head_length=0.3, lw=2.0,
                              alpha=0.2, color='black')
        # -----------------------------------------

        # plot option policy
        x_pos = np.zeros(len(opt_states))
        y_pos = np.zeros(len(opt_states))
        for i in range(1, len(opt_states)):
            x_pos = opt_states[i-1][1]
            y_pos = opt_states[i-1][0]
            dx, dy = opt_states[i][1] - x_pos, opt_states[i][0] - y_pos
            plt.arrow(x_pos, y_pos, dx, dy, length_includes_head=True,
                      head_width=0.15, head_length=0.3, lw=2.0,
                      color='black', label=r'$\pi_{TAPRL}$')

        for j in range(len(opt_states)):
            plt.scatter(opt_states[j][1], opt_states[j][0], 300,
                        color='black', alpha=0.5)

        if adjacent_states is not None:
            # remove duplicate subgoal states from adjacent states list
            for m in range(1, len(opt_states)):
                adjacent_states.remove(opt_states[m])
            # remove duplicate states, only keep the unique adjacent states
            unq_adj_states = []
            for m in range(len(adjacent_states)):
                if adjacent_states[m] not in unq_adj_states:
                    unq_adj_states.append(adjacent_states[m])

            adj_x = [unq_adj_states[i][1] for i in range(len(unq_adj_states))]
            adj_y = [unq_adj_states[i][0] for i in range(len(unq_adj_states))]
            plt.scatter(adj_x, adj_y, 250, color='white', alpha=0.25)

        # plot option policy for offline optimal policy from model
        if opt_states_sim is not None:
            x_pos_sim = np.zeros(len(opt_states_sim))
            y_pos_sim = np.zeros(len(opt_states_sim))
            for i in range(1, len(opt_states_sim)):
                x_pos_sim = opt_states_sim[i - 1][1]
                y_pos_sim = opt_states_sim[i - 1][0]
                dx_sim, dy_sim = opt_states_sim[i][1] - x_pos_sim, \
                                 opt_states_sim[i][0] - y_pos_sim
                plt.arrow(x_pos_sim, y_pos_sim, dx_sim, dy_sim,
                          ls='--', length_includes_head=True,
                          head_width=0.15, head_length=0.3, lw=2.0,
                          color='black', alpha=0.25, label=r'$\hat{\pi}^*$')
            plt.scatter(opt_states_sim[-1][0], opt_states_sim[-1][1],
                        s=400, marker='*', color='black', alpha=0.25)
        # save plot
        print('saving plot as pdf . . .')
        plt.savefig("figures/options_{}.pdf".format(time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')
        print('done!')
        plt.show()

    def animate_single_policy(self, Y, states, algo_name):
        nrows, ncols = Y.shape
        image = np.zeros(nrows * ncols)
        # Set every other cell to a random number (this would be your data)
        image = Y
        # Reshape things into a 9x9 grid.
        image = image.reshape((nrows, ncols))
        row_labels = range(nrows)
        col_labels = range(ncols)
        plt.matshow(image, fignum=0)
        plt.xticks(range(ncols), range(ncols))
        plt.yticks(range(nrows), range(nrows))
        camera = Camera(plt.figure(figsize=(10, 8)))

        # policy
        x = [states[i][1] for i in range(len(states))]
        y = [states[i][0] for i in range(len(states))]

        for i in range(len(x)):
            xdata, ydata = x[:i + 1], y[:i + 1]
            plt.plot(xdata, ydata, '-', color='white', markersize=10, lw=2.0)
            plt.plot(xdata[-1], ydata[-1], 'o', color='red', markersize=10, lw=2.0)
            # Make a Y.shape grid...
            nrows, ncols = Y.shape
            image = np.zeros(nrows * ncols)
            # Set every other cell to a random number (this would be your data)
            image = Y
            # Reshape things into a 9x9 grid.
            image = image.reshape((nrows, ncols))
            row_labels = range(nrows)
            col_labels = range(ncols)
            plt.matshow(image, fignum=0)  # matshow: gridworld
            plt.xticks(range(ncols), col_labels)
            plt.yticks(range(nrows), row_labels)
            camera.snap()

        anim = camera.animate(blit=True)
        if algo_name == 'SARSA':
            # anim.save('figures/policy_animation_SARSA_action_updated.mp4')
            anim.save('figures/policy_animation_SARSA_action_updated1.gif', writer='imagemagick')
        elif algo_name == 'Q_learning':
            # anim.save('figures/policy_animation_Qlearning_action_updated.mp4')
            anim.save('figures/policy_animation_Qlearning_action_updated1.gif', writer='imagemagick')
        plt.show()

    def animate_multiple_policy(self, opt_policy_info, algo_name, anim_type,
                                env_type, goal_type, subgoal_type):
        # reward plot
        # visualize optimal policy
        env = FourRooms(env_type, goal_type, subgoal_type)
        Y, _ = env.rewards_distribution()

        nrows, ncols = Y.shape
        image = np.zeros(nrows * ncols)
        # Set every other cell to a random number (this would be your data)
        image = Y
        # Reshape things into a 9x9 grid.
        image = image.reshape((nrows, ncols))
        row_labels = range(nrows)
        col_labels = range(ncols)
        plt.matshow(image, fignum=0)
        plt.xticks(range(ncols), range(ncols))
        plt.yticks(range(nrows), range(nrows))
        camera = Camera(plt.figure(figsize=(10, 8)))

        x, y = {}, {}
        for ep in range(len(opt_policy_info)):
            states = opt_policy_info[ep][0]
            x[ep] = [states[i][1] for i in range(len(states))]
            y[ep] = [states[i][0] for i in range(len(states))]

        for i in range(len(x[0])):
            for t in range(len(opt_policy_info)):
                xdata, ydata = x[t][:i + 1], y[t][:i + 1]
                plt.plot(xdata, ydata, '-', color='white', markersize=10, lw=2.0)
                plt.plot(xdata[-1], ydata[-1], 'o', color='red', markersize=10, lw=2.0)

            # Make a Y.shape grid...
            nrows, ncols = Y.shape
            image = np.zeros(nrows * ncols)
            # Set every other cell to a random number (this would be your data)
            image = Y
            # Reshape things into a 9x9 grid.
            image = image.reshape((nrows, ncols))
            row_labels = range(nrows)
            col_labels = range(ncols)
            plt.matshow(image, fignum=0)  # matshow: gridworld
            plt.xticks(range(ncols), col_labels)
            plt.yticks(range(nrows), row_labels)
            camera.snap()

        anim = camera.animate(blit=True)
        anim.save('figures/policy_animation_{}_{}.{}'.format(algo_name, time.strftime("%m%d-%H%M%S"), anim_type),
                  writer='imagemagick')
        plt.show()

    def plot_value_functions(self, V_func):

        V_func = V_func / np.max(V_func)
        V = V_func.reshape(11, 11)

        fig = plt.figure(figsize=(10, 8))
        row_labels = range(11)
        col_labels = range(11)
        cax = plt.matshow(V, fignum=1)
        plt.xticks(range(11), col_labels)
        plt.yticks(range(11), row_labels)
        cbar = fig.colorbar(cax)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(20)

        # plt.xticks(labelsize=20)
        # plt.yticks(labelsize=20)
        plt.savefig("../figures/value_functions_FR.pdf", dpi=1200, bbox_inches='tight')
        plt.show()

    def plot_compare_rewards(self, optimal_reward_cache_real,
                             optimal_reward_cache_sim=None):

        # get the mean and std deviation of the optimal rewards
        mean_real = np.mean(optimal_reward_cache_real, axis=0)
        std_real = np.std(optimal_reward_cache_real, axis=0)

        # plot properties
        plt.figure(figsize=(12, 8))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2.0

        x_axis = [i for i in range(len(optimal_reward_cache_real[0]))]
        # plot experimental rewards error bar
        plt.plot(x_axis, mean_real, color='blue', lw=2.0, label=r'$E[R]$ (TAPRL)')
        plt.fill_between(x_axis, mean_real + 1 * std_real, mean_real - 1 * std_real,
                         color='blue', alpha=0.1)

        if optimal_reward_cache_sim is not None:
            mean_sim = np.mean(optimal_reward_cache_sim, axis=0)
            std_sim = np.std(optimal_reward_cache_sim, axis=0)
            # plot simulated rewards: does not need error bar
            plt.plot(x_axis, mean_sim, color='red', lw=2.0, label=r'$E[R]$ (offline)')

            plt.fill_between(x_axis, mean_sim + 1 * std_sim, mean_sim - 1 * std_sim,
                             color='red', alpha=0.1)

        x_axis_label = np.arange(0, len(mean_real), 5)
        plt.xticks(x_axis_label)
        plt.legend(fontsize=25, loc='lower right')
        plt.xlabel(r'timestep', fontsize=35)
        plt.ylabel(r'reward', fontsize=35)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        plt.savefig("/home/ghost-083/Research/Codes/CDC 2021/four_room/"
                    "figures/compare_rewards_{}.pdf".format(
            time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')
        plt.show()

    def plot_multiple_init_rewards(self, rewards):
        # plot properties
        plt.figure(figsize=(12, 8))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2.0

        n_vals = [4, 7, 10]
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
        plt.legend(loc='lower right', fontsize=25)

        # ---------- save plot --------
        plt.savefig("/home/ghost-083/Research/Codes/CDC 2021/four_room/"
                    "figures/multi_init_rewards_horizon_{}.pdf".format(
            time.strftime("%m%d-%H%M%S")), dpi=1200, bbox_inches='tight')
        # -----------------------------
        plt.show()

    def plot_bounds(self, L_terms, R_terms, x_vals):
        # plot properties ------------------
        plt.figure(figsize=(12, 6))
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')
        plt.rcParams['axes.linewidth'] = 2.0
        fig, ax = plt.subplots()
        # ----------------------------------
        plt.plot(x_vals, L_terms, '--o', lw=2.0, color='red', label=r'$LHS$')
        plt.plot(x_vals, R_terms, '-x', lw=2.0, color='blue', label=r'$RHS$')
        plt.xlabel(r'$\gamma$', fontsize=20)
        plt.ylabel(r'$bound$', fontsize=20)
        plt.legend(fontsize=20)
        plt.savefig("../figures/bounds_visualization.pdf",
                    dpi=1200, bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    pass

