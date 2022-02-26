import numpy as np
import pandas as pd

# preferred sim model: length_scale = 75.0, sigma_f = 0.5, sigma_y = 10.0
# preferred real model: length_scale = 1.0, sigma_f = 1.0, sigma_y = 0.2


def build_dataset():
    # import data ---> Add column header
    col_names = ['filament_distance', 'filament_diameter', 'Loss']
    phononic_data = pd.read_csv('phononic_dataset_new.csv', names=col_names, header=None)
    simulation_data = pd.DataFrame(phononic_data)  # convert into data frame
    filament_distance = simulation_data.iloc[:, 0] + 400
    filament_diameter = simulation_data.iloc[:, 1]
    loss_values = simulation_data.iloc[:, 2]
    # convert to 2D numpy array
    X1 = filament_distance.values.reshape(68, 68)
    X2 = filament_diameter.values.reshape(68, 68)
    Y = loss_values.values.reshape(68, 68)

    # save values of X1 and X2
    np.save('../data/X1.npy', X1)
    np.save('../data/X2.npy', X2)

    return X1, X2, Y


if __name__ == "__main__":
    X1, X2, Y = build_dataset()
