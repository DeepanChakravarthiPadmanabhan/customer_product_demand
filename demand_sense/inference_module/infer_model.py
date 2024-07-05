import logging
import pandas as pd
import datetime
import lightgbm as lgb
import time

from demand_sense.inference_module.test_helper import generate_test_df
from demand_sense.feature_extractor.feature_extractor import get_processed_df
from demand_sense.inference_module.test_helper import sales_d_m_or_y

LOGGER = logging.getLogger(__name__)


def infer_model(model_file, data_file, test_date):
    """
    Estimates the sales for the entire month of the given date

    :param model_file: str, path of the model
    :param data_file: str, data path
    :param test_date: str, date of the sales report required

    :return df_all: pandas.Dataframe, sales for the entire month of test date
    """
    df_train = pd.read_csv(data_file)
    df_train["date"] = pd.to_datetime(df_train["date"])

    df_test = generate_test_df(test_date, df_train)

    df_all = pd.concat([df_train, df_test])

    df_all = get_processed_df(df_all)

    test = df_all.loc[df_all.sales.isna()]
    cols = [
        col for col in df_all.columns if col not in ["date", "sales", "year"]
    ]
    x_test = test[cols]

    best_model = lgb.Booster(model_file=model_file)
    test_preds = best_model.predict(
        x_test, num_iteration=best_model.best_iteration
    )
    df_test_preds = pd.DataFrame(test_preds, columns=["sales"])
    df_all["sales"].fillna(df_test_preds["sales"], inplace=True)

    return df_all


def check_and_generate_test_date(test_date):
    """
    Checks the test date string and converts to a specific format

    :param test_date: str, test date string in format DDMMYYYY
    :param test_date: datetime.date, test date in format YYYY-MM-DD
    """
    assert len(test_date) == 8, "Enter date in format DDMMYYYY"
    if len(test_date) == 8:
        test_date = datetime.datetime.strptime(test_date, "%d%m%Y")
    return test_date


def infer_day_sales(model_file, data_file, test_date):
    """
    Estimates the sales of a particular customer on a day

    :param model_file: str, path of the model
    :param data_file: str, data path
    :param test_date: str, date of the sales report required

    :return sales_on_day: float, total sales on a particular day
    """
    df_all = infer_model(model_file, data_file, test_date)
    df_day = sales_d_m_or_y(df_all, "daily")
    sales_on_day = df_day.loc[df_day["date"] == test_date, "sales"].values[0]
    return sales_on_day


def infer_customer_sales(model_file, data_file, test_date, customer_id):
    """
    Estimates the sales of a particular customer on a day

    :param model_file: str, path of the model
    :param data_file: str, data path
    :param test_date: str, date of the sales report required
    :param customer_id: str, customer id

    :return customer_sales_on_day: float, sales of a customer on a particular day
    """
    df_all = infer_model(model_file, data_file, test_date)
    df_customer = df_all[df_all["customer_id_" + customer_id] == 1]
    df_day = sales_d_m_or_y(df_customer, "daily")
    customer_sales_on_day = df_day.loc[
        df_day["date"] == test_date, "sales"
    ].values[0]
    return customer_sales_on_day


def infer_product_sales(model_file, data_file, test_date, product_id):
    """
    Estimates the sales of a particular product on a specific customer

    :param model_file: str, path of the model
    :param data_file: str, data path
    :param test_date: str, date of the sales report required
    :param product_id: str, product id

    :return product_sales_on_day: float, sales of a product on a day
    """
    df_all = infer_model(model_file, data_file, test_date)
    df_product = df_all[df_all["product_id_" + product_id] == 1]
    df_day = sales_d_m_or_y(df_product, "daily")
    product_sales_on_day = df_day.loc[
        df_day["date"] == test_date, "sales"
    ].values[0]
    return product_sales_on_day


def infer_customer_product_sales(
    model_file, data_file, test_date, customer_id, product_id
):
    """
    Estimates the sales of a particular product on a specific customer

    :param model_file: str, path of the model
    :param data_file: str, data path
    :param test_date: str, date of the sales report required
    :param customer_id: str, customer id
    :param product_id: str, product id

    :return customer_product_sales_on_day: float, sales of a product on a customer
    """
    df_all = infer_model(model_file, data_file, test_date)
    df_customer = df_all[df_all["customer_id_" + customer_id] == 1]
    df_customer_product = df_customer[df_customer["product_id_" + product_id] == 1]
    df_day = sales_d_m_or_y(df_customer_product, "daily")
    customer_product_sales_on_day = df_day.loc[
        df_day["date"] == test_date, "sales"
    ].values[0]
    return customer_product_sales_on_day


def infer(
    model_file="model/model.txt",
    data_file="data_trc.csv",
    test_date="20102019",
    infer_level="day",
    customer_id="S0028",
    product_id="P0268",
):
    """
    Estimates the sales statistics as requested

    :param model_file: str, path of the model
    :param data_file: str, data path
    :param test_date: str, date of the sales report required
    :param infer_level: str, type of sales statistics required
    :param customer_id: str, customer id
    :param product_id: str, product id

    :return output: float, sales statistics requested
    """
    s_time = time.time()
    test_date = check_and_generate_test_date(test_date)
    output = 0
    if infer_level == "day":
        output = infer_day_sales(model_file, data_file, test_date)
        print("Total sales on {}:{}".format(test_date, output))
    elif infer_level == "customer":
        output = infer_customer_sales(model_file, data_file, test_date, customer_id)
        print(
            "Sales in customer {} on {}: {}".format(customer_id, test_date, output)
        )
    elif infer_level == "product":
        output = infer_product_sales(
            model_file, data_file, test_date, product_id
        )
        print(
            "Sales of product {} on {}: {}".format(
                product_id, test_date, output
            )
        )
    elif infer_level == "customer_product":
        output = infer_customer_product_sales(
            model_file, data_file, test_date, customer_id, product_id
        )
        print(
            "Sales in customer {} of product {} on {}: {}".format(
                customer_id, product_id, test_date, output
            )
        )
    print("Total time taken: {} seconds".format(time.time() - s_time))
    return output
