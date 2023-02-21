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


@pytest.mark.skip(reason="Skipping due to gpu memory constaints ...")
def test_get_answers_with_extractive_reader(mock_client):
    response = mock_client.post(
        "/GetAnswersRequest",
        json={
            "reader": {"reader_id": "ExtractiveReader"},
            "queries": ["Which country is Canberra located in?"],
            "contexts": [
                [
                    "Canberra is the capital city of Australia. Founded following the federation of the colonies of Australia as the seat of government for the new nation, it is Australia's largest inland city"
                ]
            ],
        },
    )
    assert response.status_code == 201
    answers = response.json()


@pytest.mark.skip(reason="Skipping due to gpu memory constaints ...")
def test_get_answers_with_generative_reader(mock_client):
    response = mock_client.post(
        "/GetAnswersRequest",
        json={
            "reader": {"reader_id": "GenerativeFiDReader"},
            "queries": ["Which country is Canberra located in?"],
            "contexts": [
                [
                    "Canberra is the capital city of Australia. Founded following the federation of the colonies of Australia as the seat of government for the new nation, it is Australia's largest inland city"
                ]
            ],
        },
    )
    assert response.status_code == 201
    answers = response.json()
