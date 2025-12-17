import logging.config
import os
import sys

from dotenv import load_dotenv

load_dotenv()
ENV = os.getenv("environment_type", "dev")

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s"
        },
        "json": {
            "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
            "format": "%(asctime)s %(levelname)s %(name)s %(message)s %(filename)s %(lineno)d",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "stream": sys.stdout,
            "formatter": "standard",
        },
        "json_file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "filename": "prod_json_logs.log",
            "formatter": "json",
        },
    },
    "loggers": {"app": {"level": "DEBUG"}},
    "root": {
        "level": "WARNING",  # This is the logging level of all child loggers whose level is NOTSET
        "handlers": ["console"] if ENV == "dev" else ["json_file"],
    },
}


def setup_logging():
    logging.config.dictConfig(LOGGING_CONFIG)
