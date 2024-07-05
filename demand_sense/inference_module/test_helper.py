import logging
import pandas as pd
import calendar
import datetime
from itertools import product

LOGGER = logging.getLogger(__name__)


def givedays(f):
    """A function to return all the days in the month of a given date

    :param f: datetime.date parsed date in the format YYYY-MM-DD

    :return days: list, all days in the month of the given date, f
    """
    year = f.year
    month = f.month
    num_days = calendar.monthrange(year, month)[1]
    days = [datetime.date(year, month, day) for day in range(1, num_days + 1)]
    return days


def generate_test_df(test_date, prev_data):
    """
    Generate a test dataframe as a cartesian product of all days in the month
    of the test date given, customers, and products

    :param test_date: datetime.date parsed date in the format YYYY-MM-DD
    :param prev_date: pandas.Dataframe of previous month sales information

    :return df_test: pandas.Dataframe as product of days, customers, products
    """
    test_dates = givedays(test_date)
    products = prev_data["product_id"].unique()
    customers = prev_data["customer_id"].unique()
    df_test = pd.DataFrame(
        product(products, customers, test_dates),
        columns=["product_id", "customer_id", "date"],
    )
    df_test["date"] = pd.to_datetime(df_test["date"])
    return df_test


def sales_d_m_or_y(df_t, time):
    """
    Estimates sales across the periods: days, month, year

    :param df_t: pandas.Dataframe containing sales and date column
    :param time: str, time frame for sales aggregation

    :return df_t: pandas.Dataframe, aggregates sales information
    """
    df_t = df_t.copy()
    if time == "daily":
        df_t.date = df_t.date.apply(lambda x: str(x)[:-9])
    elif time == "monthly":
        df_t.date = df_t.date.apply(lambda x: str(x)[:-12])
    else:
        df_t.date = df_t.date.apply(lambda x: str(x)[:4])
    df_t = df_t.groupby("date")["sales"].sum().reset_index()
    df_t.date = pd.to_datetime(df_t.date)
    return df_t
