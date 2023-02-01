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

import pytest


@pytest.mark.skip(reason="Skipping due to absence of index ...")
def test_get_documents_with_colbert_retriever(mock_client):
    response = mock_client.post(
        "/RetrieveRequest",
        json={
            "retriever": {"retriever_id": "ColBERTRetriever"},
            "index_id": "Test Index",
            "queries": ["Which country is Canberra located in?"],
        },
    )
    assert response.status_code == 201
    documents = response.json()


@pytest.mark.skip(reason="Skipping due to absence of index ...")
def test_get_answers_with_bm25_retriever(mock_client):
    response = mock_client.post(
        "/RetrieveRequest",
        json={
            "retriever": {"retriever_id": "BM25Retriever"},
            "index_id": "Test Index",
            "queries": ["Which country is Canberra located in?"],
        },
    )
    assert response.status_code == 201
    documents = response.json()


@pytest.mark.skip(reason="Skipping due to absence of index ...")
def test_get_answers_with_dpr_retriever(mock_client):
    response = mock_client.post(
        "/RetrieveRequest",
        json={
            "retriever": {"retriever_id": "DPRRetriever"},
            "index_id": "Test Index",
            "queries": ["Which country is Canberra located in?"],
        },
    )
    assert response.status_code == 201
    documents = response.json()
