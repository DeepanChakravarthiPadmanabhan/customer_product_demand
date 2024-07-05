import pandas as pd

from demand_sense.feature_extractor.lag import lag_features
from demand_sense.feature_extractor.date_features import create_date_features
from demand_sense.feature_extractor.rolling_mean import roll_mean_features
from demand_sense.feature_extractor.expanding_mean_window import ewm_features


def get_processed_df(data):
    """
    Extracts features in the time series data

    :param data: pandas.Dataframe, time series data

    :return data: pandas.Dataframe, time series data with various features
    """

    # creates date features such as month, weekend, year, day of week
    data = create_date_features(data)
    # lag features by shifting time data
    data = lag_features(data, [91, 98, 105,
                               # 112, 119, 126, 182, 364, 546
                               ])
    # rolling mean as a window
    # data = roll_mean_features(data, [365, 546])
    # alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    # lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]
    # expanding window features across various lags and alpha values
    # data = ewm_features(data, alphas, lags)
    # encode categorical features
    data = pd.get_dummies(
        data, columns=["customer_id", "product_id", "day_of_week", "month"]
    )
    return data


def split(data):
    train = data.loc[(data["date"] < "2019-01-01"), :]
    # last three months data for validation
    val = data.loc[
        (data["date"] >= "2019-01-01") & (data["date"] < "2019-04-01"), :
    ]
    train_features = [
        col for col in train.columns if col not in ["date", "sales", "year"]
    ]
    y_train = train["sales"]
    x_train = train[train_features]
    y_val = val["sales"]
    x_val = val[train_features]
    return x_train, y_train, x_val, y_val, train_features
