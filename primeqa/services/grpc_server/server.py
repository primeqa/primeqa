import logging
import time
from concurrent import futures

import grpc

from primeqa.services.configurations import Settings
from primeqa.services.cred_helpers import get_grpc_server_credentials
from primeqa.services.grpc_server.grpc_generated import reader_pb2_grpc
from primeqa.services.grpc_server.grpc_generated import retriever_pb2_grpc
from primeqa.services.grpc_server.grpc_generated import indexer_pb2_grpc
from primeqa.services.grpc_server.grpc_generated import reranker_pb2_grpc

from primeqa.services.grpc_server.reader_service import ReaderService
from primeqa.services.grpc_server.retriever_service import RetrieverService
from primeqa.services.grpc_server.indexer_service import IndexerService
from primeqa.services.grpc_server.reranker_service import RerankerService


class GrpcServer:
    def __init__(self, config: Settings = None, logger: logging.Logger = None):
        try:
            if logger is None:
                self._logger = logging.getLogger(self.__class__.__name__)
            else:
                self._logger = logger

            # Initialize application config
            if config is None:
                self._config = Settings()
            else:
                self._config = config
        except Exception as ex:
            self._logger.exception("Error configuring server: %s", ex)
            raise

    def run(self) -> None:
        start_t = time.time()

        # Set server options
        max_conn_age_option = (
            "grpc.max_connection_age_ms",
            self._config.grpc_max_connection_age_secs * 1000,
        )
        max_conn_age_grace_option = (
            "grpc.max_connection_age_grace_ms",
            self._config.grpc_max_connection_age_grace_secs * 1000,
        )
        server_options = (
            max_conn_age_option,
            max_conn_age_grace_option,
        )
        # Start gRPC server instances
        try:
            server = grpc.server(
                futures.ThreadPoolExecutor(
                    max_workers=self._config.num_threads_per_worker
                ),
                options=server_options,
            )

            # Add reader service
            reader_pb2_grpc.add_ReadingServiceServicer_to_server(
                ReaderService(config=self._config), server
            )

            # Add index service
            indexer_pb2_grpc.add_IndexingServiceServicer_to_server(
                IndexerService(config=self._config), server
            )

            # Add retriever service
            retriever_pb2_grpc.add_RetrievingServiceServicer_to_server(
                RetrieverService(config=self._config), server
            )
            
            # Add reranker service
            reranker_pb2_grpc.add_RerankerServiceServicer_to_server(
                RerankerService(config=self._config), server
            )

            if self._config.require_ssl:
                server_credentials = get_grpc_server_credentials(
                    self._config, self._logger
                )  # TLS authentication
                server.add_secure_port(
                    f"[::]:{self._config.grpc_port}", server_credentials
                )
            else:
                server.add_insecure_port(f"[::]:{self._config.grpc_port}")

            # Start server
            server.start()
            self._logger.info(
                "Server instance started on port %s - initialization took %d seconds",
                self._config.grpc_port,
                time.time() - start_t,
            )
            server.wait_for_termination()
        except Exception as ex:
            self._logger.exception("Error starting server: %s", ex)
            raise
