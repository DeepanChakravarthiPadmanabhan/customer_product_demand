import logging
import click

from demand_sense.utils import setup_logging
from demand_sense.trainer.train_model import train_model
from demand_sense.trainer.train_model_hpo import train_model_hpo

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
)
@click.option("--log_dir", default="")
@click.option("--model_dir", default="model/")
@click.option("--data_file", default="data/prepared_sales_data.csv")
@click.option("--hpo", is_flag=True)
def train(log_level, log_dir, model_dir, data_file, hpo):
    """
    Train module

    :param log_level: str, logger level
    :param log_dir: str, specific log directory
    :param model_dir: str, model directory
    :param data_file: str, data path
    :param hpo: bool, whether to train with hyperparameter optimization using
    OPTUNA
    """
    setup_logging(log_level=log_level, log_dir=log_dir)
    LOGGER.info(">training model")
    if hpo:
        train_model_hpo(model_dir, data_file)
    else:
        train_model(model_dir, data_file)


if __name__ == "__main__":
    train()
