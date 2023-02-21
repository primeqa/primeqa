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

from typing import List
from fastapi import APIRouter, status, HTTPException

from primeqa.services.exceptions import PATTERN_ERROR_MESSAGE, Error
from primeqa.services.factories import RETRIEVERS_REGISTRY
from primeqa.services.rest_server.utils import generate_parameters
from primeqa.services.rest_server.data_models import Retriever

router = APIRouter()


@router.get(
    "/retrievers",
    status_code=status.HTTP_200_OK,
    response_model=List[Retriever],
    tags=["Retriever"],
)
def get_retrievers():
    try:
        return [
            {
                "retriever_id": retriever_id,
                "parameters": generate_parameters(
                    retriever, skip=["index_root", "index_name"]
                ),
                "engine_type": retriever.get_engine_type(),
            }
            for retriever_id, retriever in RETRIEVERS_REGISTRY.items()
        ]
    except Error as err:
        error_message = err.args[0]

        # Identify error code
        mobj = PATTERN_ERROR_MESSAGE.match(error_message)
        if mobj:
            error_code = mobj.group(1).strip()
            error_message = mobj.group(2).strip()
        else:
            error_code = 500

        raise HTTPException(
            status_code=500,
            detail={"code": error_code, "message": error_message},
        ) from None
