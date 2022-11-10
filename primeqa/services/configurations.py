import functools
import logging
import os
from argparse import ArgumentTypeError
from configparser import ConfigParser
from distutils.util import strtobool
from os import environ
from typing import Union, Callable, Type, Optional, Any, get_type_hints

from pkg_resources import resource_filename

_ENVIRONMENT_VARIABLE = "config_override_section"
_CONFIG_FILE = resource_filename("primeqa.services.config", "config.ini")


class ConfigurationError(Exception):
    def __init__(self, message):
        help_message = (
            "You can override config parameters in the config file %s, setting the "
            "'%s' environment variable to a config file target, or by setting the "
            "environment variable with the same name (environment variables override config).",
            _CONFIG_FILE,
            _ENVIRONMENT_VARIABLE,
        )
        super().__init__("%s -- %s", message, help_message)


class ConfigurationTypeError(ConfigurationError, TypeError):
    pass


class ConfigurationLookupError(ConfigurationError, LookupError):
    pass


def positive_integer_type(value):
    try:
        ivalue = int(value)
    except ValueError as ex:
        raise ArgumentTypeError(
            f"{value} is an invalid positive int value: {ex}"
        ) from ex

    if ivalue < 1:
        raise ArgumentTypeError(f"{value} is an invalid positive int value")
    return ivalue


def float_type_between_zero_and_one(value):
    try:
        fvalue = float(value)
    except ValueError as ex:
        raise ArgumentTypeError(f"{value} is an invalid float value: {ex}") from ex

    if fvalue < 0.0 or fvalue > 1.0:
        raise ArgumentTypeError(f"{value} is not in the range [0.0,1.0]")

    return fvalue


def config_value(
    property_type: Union[Callable, Type] = str,
    default: Optional[Any] = None,
    is_secret: bool = False,
) -> Callable:
    """
    Decorator factory for Settings config values.

    Example of an int parameter which will default to 42:

    ```python
    @config_value(property_type: int, default: 42)
    def foo(self): pass
    ```

    Note that if calling with default arguments it must be called as @config_value().

    :param Union[Callable, Type] property_type: called with the property string value to create final property value
    :param Optional[Any] default: default value of the parameter if no value in config, if None the value is required
    :param bool is_secret: omits sensitive (True) values from dictionary view
    :return: decorator for a config value
    :rtype: Callable
    """

    def outer_wrapper(f: Callable) -> property:
        property_type_hints = get_type_hints(property_type)
        if isinstance(property_type, type):
            return_type = property_type
        elif "return" in property_type_hints:
            return_type = property_type_hints["return"]
        else:
            return_type = Any

        @property
        @functools.wraps(f)
        def inner_wrapper(self: "Settings", *args, **kwargs) -> return_type:
            property_name = f.__name__
            return self._get_or_process_config_value(
                property_name, property_type, default, is_secret
            )

        return inner_wrapper

    return outer_wrapper


def probability_float(value) -> float:
    value = float(value)
    if 0.0 < value > 1.0:
        raise ValueError(f"Expected 0 <= value <= 1 but was {value}")
    return value


def str_or_file(value: str) -> str:
    if os.path.isfile(value):
        with open(value, "rb") as f:
            return f.read().strip()
    else:
        return value


class Settings(object):
    """
    Helper class wraps config/config.ini and returns the parameter
        settings for the section referenced in the environment variable
        for `environment`. If no such environment variable is set,
        then uses the default section of the config.

    Also checks environment variables to see if there are any overrides,
    if so, will use those instead of the values from the config file.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger

        self._config_value_cache = {}
        self._raw_config_values = self._load_settings_from_config_and_env_vars()
        self._is_config_value_secret = {}
        self._logger.debug(f"Settings loaded with value: {self}")

    @config_value(property_type=int)
    def seed(self):
        pass

    @config_value(property_type=str)
    def mode(self):
        pass

    @config_value(property_type=positive_integer_type)
    def grpc_port(self):
        pass

    @config_value(property_type=positive_integer_type)
    def num_threads_per_worker(self):
        pass

    @config_value(property_type=positive_integer_type)
    def num_grpc_server_workers(self):
        pass

    @config_value(property_type=positive_integer_type)
    def grpc_max_connection_age_secs(self):
        pass

    @config_value(property_type=positive_integer_type)
    def grpc_max_connection_age_grace_secs(self):
        pass

    @config_value(property_type=positive_integer_type)
    def readiness_request_timeout_secs(self):
        pass

    @config_value(property_type=bool)
    def require_ssl(self):
        pass

    @config_value(property_type=bool)
    def require_client_auth(self):
        pass

    @config_value(property_type=str_or_file, is_secret=True)
    def tls_ca_cert(self):
        pass

    @config_value(property_type=str_or_file, is_secret=True)
    def tls_server_cert(self):
        pass

    @config_value(property_type=str_or_file, is_secret=True)
    def tls_server_key(self):
        pass

    @config_value(property_type=str_or_file, is_secret=True)
    def tls_client_cert(self):
        pass

    @config_value(property_type=str_or_file, is_secret=True)
    def tls_client_key(self):
        pass

    @config_value(property_type=str, is_secret=True)
    def tls_override_authority(self):
        pass

    @config_value(property_type=str)
    def rest_host(self):
        pass

    @config_value(property_type=positive_integer_type)
    def rest_port(self):
        pass

    @config_value(property_type=positive_integer_type)
    def num_rest_server_workers(self):
        pass

    def _get_config_dict(self):
        config_dict = {}
        for property_name in dir(self):
            if property_name.startswith("_"):
                continue

            try:
                property_value = getattr(self, property_name)
                if self._is_config_value_secret[property_name]:
                    property_value = "*********"
            except ConfigurationLookupError:
                property_value = None

            config_dict[property_name] = property_value
        return config_dict

    def __repr__(self):
        config_dict = self._get_config_dict()
        return f"{self.__class__}({repr(config_dict)})"

    def __str__(self):
        config_dict = self._get_config_dict()
        return f"{self.__class__.__name__}({config_dict})"

    def _load_settings_from_config_and_env_vars(self):
        settings_from_config_file = ConfigParser(
            allow_no_value=True, interpolation=None
        )
        self._logger.debug("Reading default settings from %s", _CONFIG_FILE)
        settings_from_config_file.read(_CONFIG_FILE)

        if _ENVIRONMENT_VARIABLE in environ:
            environment = environ[_ENVIRONMENT_VARIABLE]
            if environment not in settings_from_config_file:
                environment = settings_from_config_file.default_section
        else:
            environment = settings_from_config_file.default_section

        self._logger.info("Initializing settings with values for %s", environment)
        settings = settings_from_config_file[environment]

        return self._update_with_environment_variable_overrides(settings)

    def _update_with_environment_variable_overrides(self, config):
        for key in config.keys():
            if key in environ:
                self._logger.debug(
                    f"Updating the default value of {key} with env var override: {environ[key]}"
                )
                config[key] = environ[key]
        return config

    def _get_or_process_config_value(
        self,
        property_name: str,
        property_type: Union[Callable, Type],
        default: Optional[Any],
        is_secret: bool,
    ) -> Any:
        # see @config_value decorator factory docstring for details
        if property_name not in self._config_value_cache:
            self._is_config_value_secret[property_name] = is_secret

            if self._raw_config_values[property_name] is None and default is None:
                raise ConfigurationLookupError(f"Missing property {property_name}")

            property_value = self._raw_config_values.get(property_name, None)
            if property_value is not None:
                try:
                    logging.debug("Attempting to construct value for %s", property_name)
                    if property_type == bool:
                        property_value = bool(
                            strtobool(property_value)
                        )  # bool('False') == True
                    elif property_type == set:
                        property_value = set(map(str.strip, property_value.split(",")))
                    else:
                        property_value = property_type(property_value)
                except Exception as ex:
                    raise ConfigurationTypeError(
                        f"Unable to convert property {property_name} with value {property_value} to type {property_type}"
                    ) from ex
            else:
                logging.debug("Using default value for %s", property_name)
                property_value = default
            self._config_value_cache[property_name] = property_value
        return self._config_value_cache[property_name]
