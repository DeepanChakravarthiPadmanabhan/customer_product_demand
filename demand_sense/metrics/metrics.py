import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error
from sklearn.metrics import mean_squared_error, r2_score


def smape(preds, target):
    """
    Estimates symmetric mean absolute percentage error

    :param preds: float array, predicted values
    :param target: float array, target values

    :return smape_val: float, smape error
    """
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val


def lgbm_smape(preds, train_data):
    """
    Estimates symmetric mean absolute percentage error from LightGBM model out

    :param preds: float array, model output predictions
    :param train_data: lgbm train data object

    :return error: tuple, label, error value, flag
    """
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return "SMAPE", smape_val, False
