import logging
import datetime
import sys
import os


def setup_logging(log_level, log_dir):
    """
    To set up logging
    :param log_level: str, log level
    :param log_dir: str, log directory
    """

    log_level = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
    }[log_level.upper()]
    root_logger = logging.getLogger()
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    root_logger.setLevel(log_level)
    root_logger.addHandler(handler)
    if log_dir:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        log_filename = datetime.datetime.now().strftime("%Y-%m-%d") + ".log"
        filehandler = logging.FileHandler(
            filename=os.path.join(log_dir, log_filename)
        )
        filehandler.setFormatter(formatter)
        root_logger.addHandler(filehandler)
