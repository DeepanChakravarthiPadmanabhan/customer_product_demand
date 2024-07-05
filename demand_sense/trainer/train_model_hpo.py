import os
import logging
import pandas as pd
import lightgbm as lgb
import numpy as np
import optuna

from demand_sense.utils.check_df import check_df
from demand_sense.feature_extractor.feature_extractor import get_processed_df
from demand_sense.feature_extractor.feature_extractor import split
from demand_sense.metrics.metrics import mean_absolute_error
from demand_sense.metrics.metrics import mean_squared_error
from demand_sense.metrics.metrics import explained_variance_score
from demand_sense.metrics.metrics import r2_score

LOGGER = logging.getLogger(__name__)


class Objective:
    def __init__(self):
        self.best_booster = None
        self._booster = None

    def __call__(self, trial, data_file):
        """
        Objective module for performing hyperparameter optimization

        :param trial: int, trial id
        :param data_file: str, data file path
        """
        data = pd.read_csv(data_file)
        data["date"] = pd.to_datetime(data["date"])
        check_df(data)
        LOGGER.info(
            "Data period: %s to %s", data["date"].min(), data["date"].max()
        )
        LOGGER.info(
            "Number of days: %s", data["date"].max() - data["date"].min()
        )
        data = get_processed_df(data)

        x_train, y_train, x_test, y_test, train_features = split(data)

        dtrain = lgb.Dataset(x_train, label=y_train)
        dvalid = lgb.Dataset(x_test, label=y_test)

        # hyperparameter optimization search space for different variables
        param = {
            "objective": "regression",
            "metric": "l1",
            "verbosity": 1,
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.005, 0.2, log=True
            ),
            "num_leaves": trial.suggest_int("num_leaves", 5, 15),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "feature_fraction": trial.suggest_uniform(
                "feature_fraction", 0.5, 1
            ),
        }

        # Prune config for HPO
        pruning_callback = optuna.integration.LightGBMPruningCallback(
            trial, "l1"
        )
        gbm = lgb.train(
            param,
            dtrain,
            valid_sets=[dvalid],
            verbose_eval=False,
            callbacks=[pruning_callback],
        )

        self._booster = gbm

        preds = gbm.predict(x_test)
        pred_labels = np.rint(preds)
        error = mean_absolute_error(y_test, pred_labels)
        return error

    def callback(self, study, trial):
        if study.best_trial == trial:
            self.best_booster = self._booster


def validate_model(data_file, best_model):
    """
    Validation module

    :param data_file: str, data file path
    :param best_model: LightGBM model after hyperparameter optimization
    """
    data = pd.read_csv(data_file)
    data["date"] = pd.to_datetime(data["date"])
    data = get_processed_df(data)
    x_train, y_train, x_val, y_val, train_features = split(data)
    LOGGER.info("%%% Validation metrics %%%")
    y_pred_val = best_model.predict(x_val)
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


def train_model_hpo(model_dir, data_file):
    """
    Trains a LightGBM model with hyperparameter optimization using OPTUNA
    and saves it in the model_dir

    :param model_dir: str, model directory
    :param data_file: str, data file path
    """
    objective = Objective()
    func = lambda trial: objective(trial, data_file)
    study = optuna.create_study(direction="minimize")
    study.optimize(func, n_trials=10, callbacks=[objective.callback])

    best_model = objective.best_booster
    print("Number of finished trials:", len(study.trials))
    print("Best trial:", study.best_trial.params)
    validate_model(data_file, best_model)
    best_model.save_model(os.path.join(model_dir, "model_trained_hpo.txt"))
