from demand_sense.feature_extractor.utils import random_noise


def lag_features(dataframe, lags):
    """
    Estimate lag features for time series data

    :param dataframe: pandas.Dataframe, time series data
    :param lags: list, lag values to shift time data

    :return dataframe: pandas.Dataframe with lag features
    """
    # then we define lag features function with added noise
    for lag in lags:
        dataframe["sales_lag_" + str(lag)] = dataframe.groupby(
            ["customer_id", "product_id"]
        )["sales"].transform(lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe
