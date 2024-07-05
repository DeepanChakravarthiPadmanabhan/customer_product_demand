from demand_sense.feature_extractor.utils import random_noise


def roll_mean_features(dataframe, windows):
    """
    Estimate rolling mean features for time series data

    :param dataframe: pandas.Dataframe, time series data
    :param windows: list, window lengths

    :return dataframe: pandas.Dataframe with rolling mean features
    """
    for window in windows:
        dataframe["sales_roll_mean_" + str(window)] = dataframe.groupby(
            ["customer_id", "product_id"]
        )["sales"].transform(
            lambda x: x.shift(1)
            .rolling(window=window, min_periods=10, win_type="triang")
            .mean()
        ) + random_noise(
            dataframe
        )
    return dataframe
