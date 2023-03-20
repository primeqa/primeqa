#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022-2023 PrimeQA Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time


import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from primeqa.services.configurations import Settings
from primeqa.services.rest_server import (
    documents,
    readers,
    reranked_documents,
    retrievers,
    indexers,
    indexes,
    answers,
    rerankers,
)

############################################################################################
#                                   API SERVER
############################################################################################
app = FastAPI(
    title="PrimeQA Service",
    version="0.11.3",
    contact={
        "name": "PrimeQA Team",
        "url": "https://github.com/primeqa/primeqa",
        "email": "primeqa@us.ibm.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

############################################################################################
#                           Reading APIs
############################################################################################
app.include_router(router=readers.router)
app.include_router(router=answers.router)

############################################################################################
#                           Indexing API
############################################################################################
app.include_router(router=indexers.router)
app.include_router(router=indexes.router)

############################################################################################
#                           Retrieving APIs
############################################################################################
app.include_router(router=retrievers.router)
app.include_router(router=documents.router)

############################################################################################
#                           Reranking APIs
############################################################################################
app.include_router(router=rerankers.router)
app.include_router(router=reranked_documents.router)


class RestServer:
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

        ############################################################################################
        #                                   API SERVER MIDDLEWARE
        ############################################################################################
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True if self._config.require_client_auth else False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        ############################################################################################
        #                                   API SERVER CONFIGURATION
        ############################################################################################
        if self._config.require_ssl:
            server_config = uvicorn.Config(
                app,
                host=self._config.rest_host,
                port=self._config.rest_port,
                workers=self._config.num_rest_server_workers,
                ssl_keyfile=self._config.tls_server_key,
                ssl_certfile=self._config.tls_server_cert,
                ssl_ca_certs=self._config.tls_ca_cert,
            )
        else:
            server_config = uvicorn.Config(
                app,
                host=self._config.rest_host,
                port=self._config.rest_port,
                workers=self._config.num_rest_server_workers,
            )

        # Create and run server
        try:
            uvicorn.Server(server_config).run()
            self._logger.info(
                "Server instance started on port %s - initialization took %s seconds",
                self._config.rest_port,
                time.time() - start_t,
            )
        except Exception as ex:
            self._logger.exception("Error starting server: %s", ex)
            raise
