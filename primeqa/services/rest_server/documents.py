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
from primeqa.services.factories import RETRIEVERS_REGISTRY, RetrieverFactory
from primeqa.services.rest_server.data_models import RetrieveRequest, Hit

router = APIRouter()

# Fetch store instance
STORE = StoreFactory.get_store()


@router.post(
    "/RetrieveRequest",
    status_code=status.HTTP_201_CREATED,
    response_model=List[List[Hit]],
    tags=["Retriever"],
)
def get_documents(request: RetrieveRequest):
    try:
        # Step 1: Load index information
        if request.index_id:
            index_root = STORE.get_index_directory_path(request.index_id)
            # Step 1.a: Check if `index_root` exists
            if not STORE.exists(index_root):
                raise Error(
                    ErrorMessages.FAILED_TO_LOCATE_INDEX.value.format(request.index_id)
                )

            # Step 1.b: Load index information
            index_information = STORE.get_index_information(index_id=request.index_id)
            if index_information[ATTR_STATUS] != IndexStatus.READY.value:
                raise Error(
                    ErrorMessages.INDEX_UNAVAILABLE_FOR_QUERYING.value.format(
                        index_information[ATTR_STATUS]
                    )
                )
        else:
            raise Error(ErrorMessages.INVALID_REQUEST.value.format("index_id"))

        # Step 2: Verify requested retriever exists
        try:
            retriever = RETRIEVERS_REGISTRY[request.retriever.retriever_id]
        except KeyError as err:
            raise Error(
                ErrorMessages.INVALID_RETRIEVER.value.format(
                    request.retriever.retriever_id,
                    ", ".join(RETRIEVERS_REGISTRY.keys()),
                )
            ) from err

        # Step 3: Match engine type of requested collection and retriever
        if (
            index_information[ATTR_CONFIGURATION][ATTR_ENGINE_TYPE]
            != retriever.get_engine_type()
        ):
            raise Error(
                ErrorMessages.MISMATCHED_ENGINE_TYPE.value.format(
                    index_information[ATTR_CONFIGURATION][ATTR_ENGINE_TYPE],
                    request.retriever.retriever_id,
                    retriever.get_engine_type(),
                )
            )

        # Step 4: Load default retriever keyword arguments
        retriever_kwargs = {
            k: v.default for k, v in retriever.__dataclass_fields__.items() if v.init
        }

        # Step 3: If parameters are provided in request then update keyword arguments used to instantiate retriever instance
        if request.retriever.parameters:
            for parameter in request.retriever.parameters:
                if parameter.parameter_id not in retriever_kwargs:
                    raise Error(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "retriever", parameter.parameter_id
                        )
                    )

                retriever_kwargs[parameter.parameter_id] = parameter.value

        # Step 5: Update index specific arguments
        retriever_kwargs["index_root"] = STORE.get_index_directory_path(
            request.index_id
        )
        retriever_kwargs["index_name"] = DIR_NAME_INDEX
        retriever_kwargs["collection"] = STORE.get_index_documents_file_path(
            index_id=request.index_id
        )
        if ATTR_CHECKPOINT in retriever_kwargs:
            retriever_kwargs[ATTR_CHECKPOINT] = STORE.get_checkpoint_path(
                index_information[ATTR_CONFIGURATION][ATTR_CHECKPOINT]
            )

        # Step 6: Create retriever instance
        try:
            instance = RetrieverFactory.get(retriever, retriever_kwargs)
        except (ValueError, TypeError) as err:
            raise Error(err.args[0]) from err

        # Step 7: Retrieve
        instance_fields = [
            k
            for k, v in instance.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]
        logging.info(
            "Applying '%s' retriever with parameters = %s for queries = %s",
            instance.__class__.__name__,
            {
                k: getattr(instance, k) if k in instance_fields else v
                for k, v in retriever_kwargs.items()
            },
            request.queries,
        )
        try:
            results = instance.predict(input_texts=request.queries, **retriever_kwargs)
            logging.info(
                "Applying '%s' retriever for queries = %s returns results = %s",
                instance.__class__.__name__,
                request.queries,
                results,
            )
        except TypeError as err:
            raise Error(
                ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                    f"{retriever.retriever_id} retriever"
                )
            ) from err

        # Step 8: Return
        hits = []
        for result_per_query in results:
            hits_per_query = []
            for hit in result_per_query:
                try:
                    document = STORE.get_index_document(
                        index_id=request.index_id, document_idx=hit[0]
                    )
                    hits_per_query.append(
                        {
                            "document": {
                                "text": document["text"],
                                "document_id": document["document_id"]
                                if "document_id" in document
                                else None,
                                "title": document["title"]
                                if "title" in document
                                else None,
                            },
                            "score": hit[1],
                        }
                    )
                except (FileNotFoundError, KeyError):
                    continue

            hits.append(hits_per_query)

        return hits

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
