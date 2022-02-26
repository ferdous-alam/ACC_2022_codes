import build_data
import numpy as np
from tqdm import tqdm
from visualization import visualizations_PnGMfg
import run_GP


class RewardModel:
    def __init__(self):
        self.X1, self.X2, self.Y = build_data.build_dataset()
        self.length_scale = 50.0
        self.sigma_f = 52.0
        self.sigma_y = 1.0

    def build_source_reward_model(self):
        """
        sim model
        -----------------------
        length_scale_sim = 15.0
        sigma_f_sim = 0.5
        sigma_y_sim = 10.0

        reward_model_sim, _ = run_GP.runGaussianProcess(length_scale_sim, sigma_f_sim, sigma_y_sim)
        reward_model_normalized_sim = reward_model_sim/np.max(reward_model_sim)
        visualizations_autonomous_mfg.plot_2D_data(X1, X2, reward_model_sim)
        -----------------------------------------------------------

        real model
        ----------------------
        length_scale_real = 52.0
        sigma_f_real = 50.0
        sigma_y_real = 1.0
        reward, _ = run_GP.runGaussianProcess(length_scale_real, sigma_f_real, sigma_y_real)

        """

        # run GP to build the reward model
        reward_model_source, _ = run_GP.runGaussianProcess(self.length_scale,
                                                           self.sigma_f,
                                                           self.sigma_y)

        # save visualization
        visualizations_PnGMfg.plot_2D_data(self.X1, self.X2, reward_model_source)

        # save reward models
        np.save('../data/source_reward_model', reward_model_source)

        return None

    def build_target_reward_model(self):

        # run GP to build the reward model
        # reward_model_source, _ = run_GP.runGaussianProcess(self.length_scale,
        #                                                    self.sigma_f,
        #                                                    self.sigma_y)
        # # to make it faster we use the saved source reward model

        reward_model_source = np.load('../data/source_reward_model.npy')
        reward_shift_x = np.roll(reward_model_source, 20, axis=0)
        reward_shift_y = np.roll(reward_shift_x, 5, axis=1)
        reward_mod = reward_shift_y

        for m in tqdm(range(1000)):
            F = np.zeros((len(self.X1), len(self.X2)))
            for i in range(len(self.X1)):
                for j in range(len(self.X2)):
                    F[i, j] = reward_mod[i, j] + np.random.normal(0, 0.5, 1)

            # visualize target reward model
            visualizations_PnGMfg.plot_2D_data(self.X1, self.X2, F, m)

            np.save('../data/target_reward_model_{}'.format(m), F)

        # save visualization

        return None


if __name__ == '__main__':
    model = RewardModel()
    # # first create the source reward model
    # model.build_source_reward_model()
    # # now create the target reward model
    model.build_target_reward_model()
