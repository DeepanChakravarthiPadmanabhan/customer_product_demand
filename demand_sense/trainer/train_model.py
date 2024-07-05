import os
import logging
import pandas as pd
import lightgbm as lgb
import numpy as np

from demand_sense.utils.check_df import check_df
from demand_sense.feature_extractor.feature_extractor import get_processed_df
from demand_sense.feature_extractor.feature_extractor import split
from demand_sense.metrics.metrics import lgbm_smape
from demand_sense.metrics.metrics import mean_absolute_error
from demand_sense.metrics.metrics import mean_squared_error
from demand_sense.metrics.metrics import explained_variance_score
from demand_sense.metrics.metrics import r2_score

LOGGER = logging.getLogger(__name__)


def train_model(model_dir, data_file):
    """
    Trains a LightGBM model and saves it in the model_dir

    :param model_dir: str, model directory
    :param data_file: str, data file path
    """
    LOGGER.info("Model directory: %s", model_dir)
    data = pd.read_csv(data_file)
    data["customer_id"] = data["customer_id"].astype(str)
    data["date"] = pd.to_datetime(data["date"])

    LOGGER.info("Data directory: %s", data_file)
    check_df(data)
    LOGGER.info(
        "Data period: %s to %s", data["date"].min(), data["date"].max()
    )
    LOGGER.info("Number of days: %s", data["date"].max() - data["date"].min())

    data = get_processed_df(data)

    x_train, y_train, x_val, y_val, train_features = split(data)

    lgb_params = {
        "num_leaves": 10,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "max_depth": 5,
        "verbose": 0,
        "num_boost_round": 1500,
        "early_stopping_rounds": 300,
        "nthread": -1,
    }
    lgbtrain = lgb.Dataset(
        data=x_train, label=y_train, feature_name=train_features
    )
    lgbval = lgb.Dataset(
        data=x_val,
        label=y_val,
        reference=lgbtrain,
        feature_name=train_features,
    )
    model = lgb.train(
        lgb_params,
        lgbtrain,
        valid_sets=[lgbtrain, lgbval],
        num_boost_round=lgb_params["num_boost_round"],
    )

    LOGGER.info("%%% Validation metrics %%%")
    y_pred_val = model.predict(x_val, num_iteration=model.best_iteration)
    LOGGER.info(
        "\tExplained variance: %s", explained_variance_score(y_val, y_pred_val)
    )
    LOGGER.info(
        "\tMean absolute error (MAE): %s",
        mean_absolute_error(y_val, y_pred_val),
    )
    LOGGER.info(
        "\tRoot Mean squared error (RMSE): %s",
        np.sqrt(mean_squared_error(y_val, y_pred_val)),
    )
    LOGGER.info("\tR2 score: %s", r2_score(y_val, y_pred_val))

    LOGGER.info("%%% Training done %%%")
    LOGGER.info("%%% Get final model %%%")

    x_train = data[train_features]
    y_train = data["sales"]
    lgb_params = {
        "metric": {"mae"},
        "num_leaves": 10,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "max_depth": 5,
        "verbose": 0,
        "nthread": -1,
        "num_boost_round": model.best_iteration,
    }
    lgbtrain_all = lgb.Dataset(
        data=x_train, label=y_train, feature_name=train_features
    )
    model = lgb.train(
        lgb_params, lgbtrain_all, num_boost_round=model.best_iteration
    )
    model.save_model(os.path.join(model_dir, "model_trained.txt"))
