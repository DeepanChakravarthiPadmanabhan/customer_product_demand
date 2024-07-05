import logging

LOGGER = logging.getLogger(__name__)


def check_df(dataframe, head=5):
    """
    Calculates dataframe info

    :param dataframe: pandas.Dataframe, dataframe to be checked
    :param head: int, number of rows to print
    """
    LOGGER.info("##################### Shape #####################")
    LOGGER.info(dataframe.shape)
    LOGGER.info("##################### Types #####################")
    LOGGER.info(dataframe.dtypes)
    LOGGER.info("##################### Head #####################")
    LOGGER.info(dataframe.head(head))
    LOGGER.info("##################### Tail #####################")
    LOGGER.info(dataframe.tail(head))
    LOGGER.info("##################### NA #####################")
    LOGGER.info(dataframe.isnull().sum())
    # LOGGER.info("##################### Quantiles #####################")
    # LOGGER.info(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
