import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GaussianProcess:
    """
    Details: Gaussian process is a way supervised algorithm that
    uses gaussian process (not gaussian random variable) to fit
    the training data. It can be used in both regression and
    classification.
      f(x) ~ GP(m(x), k(x,x')); x ---> input feature vector
      GP ---> Gaussian Process
      k(.,.) ---> kernel function/covariance function
    The kernel compares the similarity between the input data,
    various kernel can be chosen according to their characteristics,
    for example, RBF kernel makes smooth prediction
    Here, we use the RBG kernel
         k(x1, x2) = sigma_f^2 * exp(-0.5/length_scale^2 *||x1-x2||)
         ||x1-x2|| = norm/distance

    reference:
    Rasmussen, Carl Edward. ”Gaussian processes in machine learning.”
     Summer School on Machine Learning. Springer, Berlin, Heidelberg, 2003.
    """
    def __init__(self, X, Y, X_s, Y_s, length_scale,
                 sigma_f, sigma_y):
        """

        :param X: training data
        :param Y: training target
        :param X_s: test data
        :param Y_s: test target
        :param length_scale: horizontal scale length of kernel function
        :param sigma_f: vertical scale length of kernel function
        :param sigma_y: varaince of noise in the training data
        """
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_y = sigma_y
        self.X = X
        self.Y = Y
        self.X_s = X_s
        self.Y_s = Y_s

    def kernel(self, X1, X2):
        """

        :param X1: data
        :param X2: data
        :return: K: kernel function
        """
        # if X2 is None:
        #     dists = spdist.pdist(X1, metric='euclidean')
        #     K = self.sigma_f**2 * np.exp(-0.5*dists/self.length_scale)
        #     K = spdist.squareform(K)
        # else:
        #     dists = spdist.cdist(X1, X2, metric='euclidean')
        #     K = self.sigma_f ** 2 * np.exp(-0.5 * dists / self.length_scale)
        norm = np.sum(X1 ** 2, 1).reshape(
            -1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)
        K = self.sigma_f ** 2 * np.exp(-0.5 / self.length_scale ** 2 * norm)

        return K

    def posterior(self):
        """
        :return: mu_s = mean of predicted data,
                cov_s = variance of predicted data
        """
        Ky = self.kernel(self.X, self.X) +\
            self.sigma_y ** 2 * np.eye(len(self.X))
        K_s = self.kernel(self.X, self.X_s)
        K_ss = self.kernel(self.X_s, self.X_s)

        mu_s = K_s.T.dot(inv(Ky)).dot(self.Y)
        cov_s = K_ss - K_s.T.dot(inv(Ky)).dot(K_s)

        return mu_s, cov_s

    # Plot Gaussian Process
    def plotGPmean(self):
        """

        :return:
        """
        mu, cov = self.posterior()
        fig = plt.figure(figsize=(12, 10))
        ax = Axes3D(fig)
        ax.plot_trisurf(self.X[:, 0], self.X[:, 1], mu, linewidth=0.1)
        ax.scatter3D(self.X[:, 0], self.X[:, 1], self.Y, c=self.Y, alpha=0.25)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        # plt.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel(r'$ l_{xy} \ \ (\mu m)$', fontsize=28)
        ax.set_ylabel(r'$d \ \ (\mu m)$', fontsize=28)
        ax.set_zlabel(r'$\mathcal{R}$', fontsize=28)
        # plt.title('Gaussian Process Regression', fontsize=20)
        plt.savefig("../figures/GP_mean.pdf", dpi=1200, bbox_inches='tight')
        plt.show()

    def plotGPcov(self):
        """

        """
        mu, cov = self.posterior()
        zVal = 1.96
        fig = plt.figure(figsize=(12, 10))
        ax: Axes3D = Axes3D(fig)
        ax.plot_trisurf(self.X[:, 0], self.X[:, 1], mu + zVal * np.sqrt(np.diag(cov)), linewidth=0.1)
        ax.plot_trisurf(self.X[:, 0], self.X[:, 1], mu, linewidth=0.1)
        ax.plot_trisurf(self.X[:, 0], self.X[:, 1], mu - zVal * np.sqrt(np.diag(cov)), linewidth=0.1)
        ax.xaxis.labelpad = 20
        ax.yaxis.labelpad = 20
        ax.zaxis.labelpad = 20
        plt.legend(fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel('$filament distance (\mu m)$', fontsize=20)
        ax.set_ylabel('$filament diameter (\mu m)$', fontsize=20)
        ax.set_zlabel('loss_GP', fontsize=20)
        plt.title('Gaussian Process Regression', fontsize=20)
        ax.scatter3D(self.X[:, 0], self.X[:, 1], self.Y, c=self.Y,
                     label='training data', alpha=0.1)
        plt.show()

    def plotGPstochastic(self, meanVal):
        """

        :return:
        """
        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        fig = plt.figure(figsize=(12, 10))
        ax = Axes3D(fig)
        ax.plot_trisurf(self.X[:, 0], self.X[:, 1], meanVal, linewidth=0.1)
        ax.xaxis.labelpad = 30
        ax.yaxis.labelpad = 30
        ax.zaxis.labelpad = 30
        ax.tick_params(axis='both', which='major', labelsize=25)
        ax.set_xlabel(r'$l_{xy}$', fontsize=65)
        ax.set_ylabel(r'$d$', fontsize=65)
        ax.set_zlabel(r'$\hat{R}$', fontsize=65)
        # plt.title(r'Gaussian Process Regression', fontsize=20)
        # ax.scatter3D(self.X[:, 0], self.X[:, 1], self.Y, c=self.Y,
        #              label='training data', alpha=0.1)
        # plt.title(r'Temporal abstraction', fontsize=28)
        # save as physics model or stochastic model
        plt.savefig("figures/stochastic_model.pdf", dpi=1200, bbox_inches='tight')

        plt.show()
