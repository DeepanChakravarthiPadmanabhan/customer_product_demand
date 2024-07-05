import numpy as np


def random_noise(data):
    """
    :param data: pandas.Dataframe, time series data

    :return data: numpy array of noise
    """
    # add random noise to our dataset
    return np.random.normal(scale=1.6, size=(len(data),))
