import logging
from typing import Union, Tuple

import grpc

from primeqa.services.configurations import Settings


def get_grpc_server_credentials(
    config: Settings, logger: Union[logging.Logger, None] = None
) -> grpc.ServerCredentials:
    """
    Loads the certificate files and returns the gRPC server credentials object
    :param Settings config: application config
    :param Union[logging.Logger, None] logger: logger for logging, will create logger with same name as function otherwise
    :return: server_credentials
    :rtype: grpc.ServerCredentials
    """
    if logger is None:
        logger = logging.getLogger(get_grpc_server_credentials.__name__)
    logger.info("Loading server credentials")
    server_credentials = grpc.ssl_server_credentials(
        [
            (
                bytes(config.tls_server_key, encoding="utf-8"),
                bytes(config.tls_server_cert, encoding="utf-8"),
            )
        ],
        root_certificates=bytes(config.tls_ca_cert, encoding="utf-8"),
        require_client_auth=config.require_client_auth,
    )
    return server_credentials


def get_grpc_client_credentials(
    config: Settings, logger: Union[logging.Logger, None] = None
) -> grpc.ChannelCredentials:
    """
    Loads the certificate files and returns the gRPC client credentials object
    :param Settings config: application config
    :param Union[logging.Logger, None] logger: logger for logging, will create logger with same name as function otherwise
    :return: client_credentials
    :rtype: grpc.ChannelCredentials
    """
    if logger is None:
        logger = logging.getLogger(get_grpc_client_credentials.__name__)
    logger.info("Loading client channel credentials")
    client_credentials = grpc.ssl_channel_credentials(
        private_key=bytes(config.tls_client_key, encoding="utf-8"),
        certificate_chain=bytes(config.tls_client_cert, encoding="utf-8"),
        root_certificates=bytes(config.tls_server_cert, encoding="utf-8"),
    )
    return client_credentials


def get_grpc_target_name_override(config: Settings) -> Tuple[str, str]:
    """
    :return: SSL Target Name Override for use in grpc secure channel options arg
    :rtype: Tuple[str, str]
    """
    return "grpc.ssl_target_name_override", config.tls_override_authority
