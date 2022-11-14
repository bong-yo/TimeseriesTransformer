import logging


DEFAULT_LOG_FORMAT = "[%(asctime)s][%(filename)s]- %(levelname)s: %(message)s"
logger = logging.getLogger("data_eval")
logging.basicConfig(
    level=logging.DEBUG,
    format=DEFAULT_LOG_FORMAT,
    datefmt='%H:%M:%S',
)
