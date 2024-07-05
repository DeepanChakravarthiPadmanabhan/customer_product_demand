from demand_sense.feature_extractor.utils import random_noise


def ewm_features(dataframe, alphas, lags):
    """
    Estimate expanding window mean features for time series data

    :param dataframe: pandas.Dataframe, time series data
    :param alphas: list, alpha values for expanding window mean calculation
    :param lags: list, lag values to shift time data

    :return dataframe: pandas.Dataframe with expanding window mean features
    """
    for alpha in alphas:
        for lag in lags:
            dataframe[
                "sales_ewm_alpha_"
                + str(alpha).replace(".", "")
                + "_lag_"
                + str(lag)
            ] = dataframe.groupby(["customer_id", "product_id"])[
                "sales"
            ].transform(
                lambda x: x.shift(lag).ewm(alpha=alpha).mean()
            )
    return dataframe
