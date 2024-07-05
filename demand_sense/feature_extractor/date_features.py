def create_date_features(df):
    """
    Extracts date features for time series data

    :param df: pandas.Dataframe, time series data

    :return df: pandas.Dataframe, input dataframe with date features
    """
    df["month"] = df.date.dt.month
    df["day_of_month"] = df.date.dt.day
    df["day_of_year"] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.isocalendar().week
    df["week_of_year"] = df["week_of_year"].astype(int)
    df["day_of_week"] = df.date.dt.dayofweek
    df["year"] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df["is_month_start"] = df.date.dt.is_month_start.astype(int)
    df["is_month_end"] = df.date.dt.is_month_end.astype(int)
    return df
