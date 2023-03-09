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
    ATTR_INDEX_ID,
    ATTR_METADATA,
    ATTR_CONFIGURATION,
    ATTR_ENGINE_TYPE,
    ATTR_CHECKPOINT,
    IndexStatus,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.factories import INDEXERS_REGISTRY, IndexerFactory
from primeqa.services.rest_server.data_models import (
    IndexInformation,
    GenerateIndexRequest,
)

router = APIRouter()

# Fetch store instance
STORE = StoreFactory.get_store()


@router.post(
    "/indexes",
    status_code=status.HTTP_201_CREATED,
    response_model=IndexInformation,
    tags=["Indexer"],
)
def generate_index(request: GenerateIndexRequest):
    try:
        # Step 1: Assign unique index id
        index_information = {
            ATTR_INDEX_ID: STORE.generate_index_uuid(),
            ATTR_STATUS: IndexStatus.INDEXING.value,
            ATTR_CONFIGURATION: {},
        }

        # Step 2: Verify requested indexer
        try:
            indexer = INDEXERS_REGISTRY[request.indexer.indexer_id]
        except KeyError as err:
            raise Error(
                ErrorMessages.INVALID_INDEXER.value.format(
                    request.indexer.indexer_id,
                    ", ".join(INDEXERS_REGISTRY.keys()),
                )
            ) from err

        # Step 3: Remove existing index if index_id is provide in the request
        if request.index_id:
            STORE.delete_index(request.index_id)
            index_information[ATTR_INDEX_ID] = request.index_id

        # Step 4: Load default retriever keyword arguments
        indexer_kwargs = {
            k: v.default for k, v in indexer.__dataclass_fields__.items() if v.init
        }

        # Step 5: If parameters are provided in request then update keyword arguments used to instantiate indexer instance
        if request.indexer.parameters:
            for parameter in request.indexer.parameters:
                if parameter.parameter_id not in indexer_kwargs:
                    raise Error(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "indexer", parameter.parameter_id
                        )
                    )

                indexer_kwargs[parameter.parameter_id] = parameter.value

                # Process `checkpoint` parameter
                if parameter.parameter_id == "checkpoint":
                    # Add `checkpoint` parameter value to index information
                    index_information[ATTR_CONFIGURATION][
                        ATTR_CHECKPOINT
                    ] = indexer_kwargs["checkpoint"]

                    # Re-map checkpoint kwarg to point to checkpoint file path in the service's store
                    indexer_kwargs["checkpoint"] = STORE.get_checkpoint_path(
                        indexer_kwargs["checkpoint"]
                    )

        # Step 6: Update index specific arguments
        indexer_kwargs["index_root"] = STORE.get_index_directory_path(
            index_information[ATTR_INDEX_ID]
        )
        indexer_kwargs["index_name"] = DIR_NAME_INDEX

        # Step 7: Create indexer instance
        try:
            instance = IndexerFactory.get(indexer, indexer_kwargs)
        except (ValueError, TypeError) as err:
            raise Error(err.args[0]) from err

        # Step 8: Save index information
        # Step 8.a: Add "engine_type"  to index information
        index_information[ATTR_CONFIGURATION][
            ATTR_ENGINE_TYPE
        ] = instance.get_engine_type()

        # Step 8.b: If "metadata" is provided, add to index information
        if request.metadata:
            index_information[ATTR_METADATA] = request.metadata

        STORE.save_index_information(
            index_id=index_information[ATTR_INDEX_ID],
            information=index_information,
        )

        # Step 9: Save documents used in index
        STORE.save_index_documents(
            index_id=index_information[ATTR_INDEX_ID],
            documents=request.documents,
        )

        # Step 10: Kick-off async index generation
        try:
            instance.index(
                STORE.get_index_documents_file_path(
                    index_id=index_information[ATTR_INDEX_ID]
                ),
            )

            # Step 10.b: Set index status to "READY" once indexing is complete
            index_information[ATTR_STATUS] = IndexStatus.READY.value
        except (TypeError, RuntimeError) as err:
            index_information[ATTR_STATUS] = IndexStatus.CORRUPT.value
            logging.exception(
                "Generation failed for index with id=%s. Resultant index may be corrupted.",
                index_information[ATTR_INDEX_ID],
            )
            logging.exception(err.args[0])

        STORE.save_index_information(
            index_information[ATTR_INDEX_ID], information=index_information
        )

        # Step 11: Return
        return index_information

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


@router.get(
    "/indexes",
    status_code=status.HTTP_200_OK,
    response_model=List[IndexInformation],
    tags=["Indexer"],
)
def get_indexes(engine_type: str = None):
    # Step 1: Iterate over each index individual to return matching indexes
    resp = []
    for index_id in STORE.get_index_ids():
        try:
            # Step 1.a: Load index information from store
            index_information_dict = STORE.get_index_information(index_id=index_id)

            # Step 1.b: Skip index if engine type is provided in request and doesn't match with the one in current index's information
            if engine_type and (
                index_information_dict[ATTR_CONFIGURATION][ATTR_ENGINE_TYPE]
                != engine_type
            ):
                continue

            # Step 2.c: Place index information response payload object
            index_information_rest_response_object = {ATTR_INDEX_ID: index_id}

            # Step 2.d: Add "configuration" information
            index_information_rest_response_object[
                ATTR_CONFIGURATION
            ] = index_information_dict[ATTR_CONFIGURATION]

            # Step 2.e: Add "metadata" information if exists
            if (
                ATTR_METADATA in index_information_dict
                and index_information_dict[ATTR_METADATA]
            ):
                index_information_rest_response_object[
                    ATTR_METADATA
                ] = index_information_dict[ATTR_METADATA]

            # Step 2.f: Add "status" information
            try:
                index_information_rest_response_object[
                    ATTR_STATUS
                ] = index_information_dict[ATTR_STATUS]

            except KeyError:
                index_information_rest_response_object[
                    ATTR_STATUS
                ] = IndexStatus.CORRUPT
        except FileNotFoundError:
            logging.warning(
                ErrorMessages.FAILED_TO_LOCATE_INDEX_INFORMATION.value.format(
                    index_id
                ).strip()
            )

        resp.append(index_information_rest_response_object)

    return resp


@router.get(
    "/indexes/{index_id}/status",
    status_code=status.HTTP_200_OK,
    response_model=dict,
    tags=["Indexer"],
)
def get_index_status(index_id: str):
    try:
        return {
            ATTR_STATUS: STORE.get_index_information(index_id=index_id)[ATTR_STATUS]
        }
    except KeyError:
        return {ATTR_STATUS: IndexStatus.CORRUPT.value}
    except FileNotFoundError:
        return {ATTR_STATUS: IndexStatus.DOES_NOT_EXISTS.value}
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
