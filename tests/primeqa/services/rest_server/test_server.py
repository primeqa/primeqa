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
from typing import List
from unittest.mock import MagicMock
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
# from primeqa.services.rest_server.server import FastAPI
import sys
import os
from primeqa.services.configurations import Settings
from primeqa.services.rest_server.data_models import Reader
from primeqa.services.rest_server.server import app


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

    def test_get_readers(self, client):
        response = client.get("/readers")
        assert response.status_code == 200

    def test_get_answers(self, client):
        response = client.post(
            "/answers",
            json={
                "question": "Test",
                "contexts": [["Example text with test word in a context list"]],
                "queries": ["Example text queries prop"],
                "reader": {
                    "reader_id": "ExtractiveReader",
                    "parameters": []
                }
            },
        )
        assert response.status_code == 201, response.text

    def test_answer_with_empty_question(self, client):
        response = client.post(
            "/answer",
            json={
                "question": "",
                "retriever": {"retriever_id": "test retriever"},
                "collection": {"collection_id": "test collection"},
                "reader": {"reader_id": "test reader"},
            },
        )
        assert response.status_code == 404, response.text
        assert response.json() == {"detail": "Not Found"}

    def test_get_indexers(self, client):
        response = client.get("/indexers")
        assert response.status_code == 200, response.text

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
