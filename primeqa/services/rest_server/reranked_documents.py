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
import json
import logging
from fastapi import APIRouter, status, HTTPException

from primeqa.services.exceptions import PATTERN_ERROR_MESSAGE, Error, ErrorMessages
from primeqa.services.constants import (
    ATTR_STATUS,
    ATTR_CONFIGURATION,
    ATTR_ENGINE_TYPE,
    ATTR_CHECKPOINT,
    IndexStatus,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.factories import RERANKERS_REGISTRY, RerankerFactory
from primeqa.services.rest_server.data_models import RerankRequest, Hit

router = APIRouter()

# Fetch store instance
STORE = StoreFactory.get_store()


@router.post(
    "/RerankRequest",
    status_code=status.HTTP_201_CREATED,
    response_model=List[List[Hit]],
    tags=["Reranker"],
)
def rerank_documents(request: RerankRequest):
    try:
        
        # Step 1: Verify requested reranker exists
        try:
            reranker = RERANKERS_REGISTRY[request.reranker.reranker_id]
        except KeyError as err:
            raise Error(
                ErrorMessages.INVALID_RERANKER.value.format(
                    request.reranker.reranker_id,
                    ", ".join(RERANKERS_REGISTRY.keys()),
                )
            ) from err

        # Step 2: Load default reranker keyword arguments
        reranker_kwargs = {
            k: v.default for k, v in reranker.__dataclass_fields__.items() if v.init
        }

        # Step 3: If parameters are provided in request then update keyword arguments used to instantiate reranker instance
        if request.reranker.parameters:
            for parameter in request.reranker.parameters:
                if parameter.parameter_id not in reranker_kwargs:
                    raise Error(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "reranker", parameter.parameter_id
                        )
                    )

                reranker_kwargs[parameter.parameter_id] = parameter.value

        # Step 4: Create reranker instance
        try:
            instance = RerankerFactory.get(reranker, reranker_kwargs)
        except (ValueError, TypeError) as err:
            raise Error(err.args[0]) from err

        # Step 5: Rerank
        instance_fields = [
            k
            for k, v in instance.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]
        logging.info(
            "Applying '%s' reranker with parameters = %s for queries = %s",
            instance.__class__.__name__,
            {
                k: getattr(instance, k) if k in instance_fields else v
                for k, v in reranker_kwargs.items()
            },
            request.queries,
        )
        try:
            request_dict = json.loads(request.json())
            queries = request_dict["queries"]
            documentsperquery = request_dict["hitsperquery"]
            results = instance.predict(queries=queries, documents=documentsperquery, **reranker_kwargs)
            logging.info(
                "Applying '%s' reranker for queries = %s returns results = %s",
                instance.__class__.__name__,
                request.queries,
                results,
            )
        except TypeError as err:
            raise Error(
                ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                    f"{reranker.reranker_id} reranker"
                )
            ) from err

        # Step 8: Return
        return results

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
