#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright 2022 PrimeQA Team
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

import sys
import os
import json
import pytest

from unittest.mock import MagicMock
from fastapi.testclient import TestClient

from primeqa.services.rest_server.server import app

from primeqa.services.rest_server.utils import generate_parameters

from primeqa.pipelines.components.reader.extractive import ExtractiveReader
from primeqa.pipelines.components.reader.generative import GenerativeReader
from primeqa.pipelines.components.retriever.dense import ColBERTRetriever
from primeqa.pipelines.components.retriever.sparse import BM25Retriever
from primeqa.pipelines.components.indexer.dense import ColBERTIndexer
from primeqa.pipelines.components.indexer.sparse import BM25Indexer
from primeqa.services.factories import ReaderFactory


class TestServer:

    @pytest.fixture()
    def client(self):
        return TestClient(app)

    @pytest.fixture()
    def mock_STORE(self, mocker) -> MagicMock:
        return mocker.patch(
            "primeqa.services.rest_server.server.STORE",
            autospec=True,
        )

    @pytest.fixture()
    def mock_READERS_REGISTRY_FACTORY(self, mocker) -> MagicMock:
        return mocker.patch(
            "primeqa.services.rest_server.server.READERS_REGISTRY_FACTORY",
            autospec=True,
        )

    @pytest.fixture()
    def mock_READER_FACTORY(self, mocker) -> MagicMock:
        return mocker.patch(
            "primeqa.services.rest_server.server.READER_FACTORY",
            autospec=True,
        )

    @pytest.fixture()
    def mock_RETRIEVERS_REGISTRY_FACTORY(self, mocker) -> MagicMock:
        return mocker.patch(
            "primeqa.services.rest_server.server.RETRIEVERS_REGISTRY_FACTORY",
            autospec=True,
        )

    @pytest.fixture()
    def mock_INDEXERS_REGISTRY_FACTORY(self, mocker) -> MagicMock:
        return mocker.patch(
            "primeqa.services.rest_server.server.INDEXERS_REGISTRY_FACTORY",
            autospec=True,
        )

    def test_get_readers(self, client, mock_READERS_REGISTRY_FACTORY):
        mock_READERS_REGISTRY_FACTORY.items.return_value = [('ExtractiveReader', ExtractiveReader),
                                                            ('GenerativeReader', GenerativeReader)]
        response = client.get("/readers")
        assert response.status_code == 200, response.text
        assert response.json() == [
            {"reader_id": reader_id, "parameters": generate_parameters(reader)}
            for reader_id, reader in mock_READERS_REGISTRY_FACTORY.items()
        ]

    def test_get_answers(self, client, mock_READERS_REGISTRY_FACTORY, mock_READER_FACTORY):
        # Generate reader parameters by default
        gen_params = generate_parameters(ExtractiveReader)

        # Simulate request values
        request = {
            "contexts": [["string"]],
            "queries": ["string"],
            "reader": {
                "reader_id": "ExtractiveReader",
                "parameters": gen_params
            }
        }

        # If parameters are provided in request then update keyword arguments used to instantiate reader instance
        reader_kwargs = {}
        for parameter in request["reader"]["parameters"]:
            reader_kwargs[parameter["parameter_id"]] = parameter["value"]
        instance = ReaderFactory.get(ExtractiveReader, reader_kwargs)

        # Run "apply" per query && Add answers for current query into response object
        answers_response = []
        for idx, query in enumerate(request["queries"]):
            predictions = instance.apply(
                input_texts=[query] * len(request["contexts"][idx]),
                context=[[text] for text in request["contexts"][idx]],
            )
            answers_response.append(
                [
                    [
                        {
                            "text": prediction["span_answer_text"],
                            "start_char_offset": prediction[
                                "span_answer"
                            ]["start_position"],
                            "end_char_offset": prediction[
                                "span_answer"
                            ]["end_position"],
                            "confidence_score": prediction[
                                "confidence_score"
                            ],
                            "context_index": int(
                                prediction["example_id"]
                            ),
                        }
                        for prediction in predictions_for_context
                    ]
                    for predictions_for_context in predictions
                ]
            )

        # Return
        a0 = answers_response[0]
        mock_READERS_REGISTRY_FACTORY.return_value = a0

        response = client.post(
            "/answers",
            json=request,
        )
        assert response.status_code == 201, response.text
        assert response.json() == mock_READERS_REGISTRY_FACTORY.return_value

    def test_answer_with_empty_queries(self, client):
        # Generate reader parameters by default
        gen_params = generate_parameters(ExtractiveReader)

        # Simulate request values
        request = {
            "contexts": [["string"]],
            "queries": [],
            "reader": {
                "reader_id": "ExtractiveReader",
                "parameters": gen_params
            }
        }

        response = client.post(
            "/answers",
            json=request,
        )
        assert response.status_code == 500, response.text
        assert response.json() == {"detail": {"code": "E4005",
                                              "message": "If contexts are provided, number of contexts(1) must match number of queries(0)"}}

    def test_get_indexers(self, client, mock_INDEXERS_REGISTRY_FACTORY):
        mock_INDEXERS_REGISTRY_FACTORY.items.return_value = [('ColBERTIndexer', ColBERTIndexer),
                                                             ('BM25Indexer', BM25Indexer)]
        response = client.get("/indexers")
        assert response.status_code == 200, response.text
        assert response.json() == [
            {"indexer_id": indexer_id, "parameters": generate_parameters(indexer)}
            for indexer_id, indexer in mock_INDEXERS_REGISTRY_FACTORY.items()
        ]

    def test_generate_index(self, client):
        response = client.post(
            "/indexes",
            json={
                "indexer": {"indexer_id": "ColBERTIndexer", "parameters": [
                    {"parameter_id": "",
                     "name": None,
                     "description": None,
                     "type": None,
                     "value": None,
                     "options": None,
                     "range": None}
                ]},
                "documents": [],
                "index_id": None
            },
        )
        assert response.status_code == 201, response.text

    def test_generate_index_invalid_indexer_parameter_error(self, client):
        response = client.post(
            "/indexes",
            json={
                "indexer": {"indexer_id": "ColBERTIndexer", "parameters": [
                    {"parameter_id": "",
                     "name": None,
                     "description": None,
                     "type": None,
                     "value": None,
                     "options": None,
                     "range": None}
                ]},
                "documents": [],
                "index_id": None
            },
        )
        assert response.status_code == 500, response.text
        assert response.json() == {"detail": {"code": "E3001",
                                              "message": "Invalid indexer parameter: . Only pre-defined parameters can be modified."}}

    def test_generate_index_empty_body_error(self, client):
        response = client.post(
            "/indexes",
            json={"indexer": {},
                  "documents": [],
                  "index_id": None
                  },
        )
        assert response.status_code == 422, response.text
        assert response.json() == {"detail": [
            {"loc": ["body", "indexer", "indexer_id"], "msg": "field required", "type": "value_error.missing"}]}

    def test_get_index_status(self, client):
        index_id = "ColBERTIndexer"
        response = client.get("/index/" + index_id + "/status")
        assert response.status_code == 200, response.text

    def test_get_index_status_invalid_index(self, client):
        index_id = ""
        response = client.get("/index/" + index_id + "/status")
        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "Not Found"}

    def test_get_retrievers(self, client):
        response = client.get("/retrievers")
        assert response.status_code == 200, response.text

    def test_get_documents(self, client):
        response = client.post(
            "/documents",
            json={
                "retriever": {"retriever_id": "ColBERTRetriever"},
                "queries": [],
                "index_id": ""
            },
        )
        assert response.status_code == 201, response.text

    def test_get_documents_invalid_retriever(self, client):
        response = client.post(
            "/documents",
            json={
                "retriever": {"retriever_id": "r"},
                "queries": [],
                "index_id": "0"
            },
        )
        assert response.status_code == 500, response.text
        assert response.json() == {'detail': {'code': 'E5001',
                                              'message': 'Invalid retriever: r. Please select one of the '
                                                         'following pre-defined retrievers: ColBERTRetriever, '
                                                         'BM25Retriever'}} != {
                   'detail': {'code': 'E6002', 'message': "Index with id 0 doesn't exist."}}

    def test_get_documents_invalid_index(self, client):
        response = client.post(
            "/documents",
            json={
                "retriever": {"retriever_id": "ColBERTRetriever"},
                "queries": [],
                "index_id": "0"
            },
        )
        assert response.status_code == 500, response.text
        assert response.json() == {"detail": {"code": "E6002", "message": "Index with id 0 doesn't exist."}}
