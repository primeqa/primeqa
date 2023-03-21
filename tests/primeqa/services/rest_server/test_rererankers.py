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


def test_get_rerankers(mock_client):
    response = mock_client.get(
        "/rerankers",
    )
    assert response.status_code == 200
    rerankers = response.json()
    assert len(rerankers) == 2
    assert ["SeqClassificationReranker", "ColBERTReranker"] == [
        reranker["reranker_id"] for reranker in rerankers
    ]
