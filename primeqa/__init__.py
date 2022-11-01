import os
import logging.config
from distutils.util import strtobool
from pkg_resources import resource_filename

# Configure logger
log_config_file = resource_filename(
    "primeqa.logger",
    "verbose_logging_config.ini"
    if strtobool(os.getenv("VERBOSE", "False"))
    else "logging_config.ini",
)
logging.config.fileConfig(log_config_file)
