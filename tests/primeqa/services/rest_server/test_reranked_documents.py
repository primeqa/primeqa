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


def test_rerank_with_seq_classification_reranker(mock_client):
    response = mock_client.post(
        "/RerankRequest",
        json={
            "reranker": {"reranker_id": "SeqClassificationReranker"},
            "parameters": [
                {
                "parameter_id": "model",
                "value": "ibm/re2g-reranker-nq",
                }
            ],
            "queries": ["Which country is Canberra located in?"],
            "hitsperquery": [
                [
                    {
                    "document": {
                        "text": "A man is eating food.",
                        "document_id": "0",
                        "title": ""
                    },
                    "score": 1.4
                    },
                    {
                    "document": {
                        "text": "Someone in a gorilla costume is playing a set of drums.",
                        "document_id": "1",
                        "title": ""
                    },
                    "score": 1.4
                    },
                    {
                    "document": {
                        "text": "A monkey is playing drums.",
                        "document_id": "2",
                        "title": ""
                    },
                    "score": 1.4
                    },
                    {
                    "document": {
                        "text": "A man is riding a white horse on an enclosed ground.",
                        "document_id": "3",
                        "title": ""
                    },
                    "score": 1.4
                    }
                ]
            ]
        },
    )
    assert response.status_code == 201



