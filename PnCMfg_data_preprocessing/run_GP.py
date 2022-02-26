import numpy as np
import pandas as pd
from gaussian_process_regression import *


def runGaussianProcess(length_scale, sigma_f, sigma_y):
    """


    :return:
    """
    print('\n building gaussian model . . . \n')
    col_names = ['filament_distance', 'filament_diameter', 'Loss']
    phononic_data = pd.read_csv('phononic_dataset_new.csv', names=col_names, header=None)
    simulation_data = pd.DataFrame(phononic_data)

    filament_dis = simulation_data.iloc[:, 0]+400
    filament_dia = simulation_data.iloc[:, 1]
    loss = simulation_data.iloc[:, 2]
    ###############################################
    lxy = filament_dis.values.reshape(68, 68)
    d = filament_dia.values.reshape(68, 68)
    loss = loss.values.reshape(68, 68).ravel()
    X_train = np.c_[lxy.ravel(), d.ravel()]
    Y_train = 80 - loss

    rx, ry = np.arange(700, 1040, 5.0), np.arange(300, 640, 5.0)
    gx, gy = np.meshgrid(rx, ry)

    X = np.c_[gx.ravel(), gy.ravel()]
    gpr = GaussianProcess(X_train, Y_train, X_train, Y_train,
                          length_scale, sigma_f, sigma_y)
    # plot GP mean
    gpr.plotGPmean()
    # GP posterior
    mu, cov = gpr.posterior()

    lossStochastic = []
    for idx1 in range(len(X)):
        val = np.random.normal(mu[idx1], cov[idx1, idx1], 1)
        lossStochastic.append(val)
    lossStochastic = np.array(lossStochastic)
    lossStochastic = lossStochastic[:, -1]
    # plot GP
    # gpr.plotGPstochastic(lossStochastic)
    Y = lossStochastic.reshape(68, 68)
    return Y, mu


if __name__ == "__main__":
    runGaussianProcess(1, 1, 0.2)

