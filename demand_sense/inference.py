import logging
import click

from demand_sense.utils import setup_logging
from demand_sense.inference_module.infer_model import infer

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option(
    "--log_level",
    default="INFO",
    type=click.Choice(["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"]),
)
@click.option("--log_dir", default="")
@click.option("--model_file", default="model/model.txt")
@click.option("--data_file", default="data_trc.csv")
@click.option("--test_date", default="22102019")
@click.option(
    "--infer_level",
    default="day",
    type=click.Choice(["day", "customer", "product", "customer_product"]),
)
@click.option("--customer_id", default="S0028")
@click.option("--product_id", default="P0268")
def inference(
    log_level,
    log_dir,
    model_file,
    data_file,
    test_date,
    infer_level,
    customer_id,
    product_id,
):
    """
    Inference module

    :param log_level: str, logger level
    :param log_dir: str, specific log directory
    :param model_file: str, trained model file
    :param data_file: str, data path
    :param test_date: str, date for sales estimates in the format DDMMYYYY
    :param infer_level: str, type of sales statistics required
    :param customer_id: str, customer id
    :param product_id: str, product id
    """
    setup_logging(log_level=log_level, log_dir=log_dir)
    LOGGER.info(">inference model")
    infer(model_file, data_file, test_date, infer_level, customer_id, product_id)


if __name__ == "__main__":
    inference()
