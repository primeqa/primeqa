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


def test_get_retrievers(mock_client):
    response = mock_client.get(
        "/retrievers",
    )
    assert response.status_code == 200
    retrievers = response.json()
    assert len(retrievers) == 3
    assert ["ColBERTRetriever", "DPRRetriever", "BM25Retriever"] == [
        retriever["retriever_id"] for retriever in retrievers
    ]
