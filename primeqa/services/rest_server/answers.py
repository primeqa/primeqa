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
from primeqa.services.factories import READERS_REGISTRY, ReaderFactory
from primeqa.components.reader.extractive import ExtractiveReader
from primeqa.components.reader.generative import GenerativeFiDReader
from primeqa.services.rest_server.data_models import GetAnswersRequest, Answer

router = APIRouter()


@router.post(
    "/GetAnswersRequest",
    status_code=status.HTTP_201_CREATED,
    response_model=List[List[Answer]],
    tags=["Reader"],
)
def get_answers(request: GetAnswersRequest):
    try:
        # Step 1: If contexts are provided, number of contexts need to match number of queries
        if request.contexts and len(request.queries) != len(request.contexts):
            raise Error(
                ErrorMessages.MISSING_CONTEXT.value.format(
                    len(request.contexts), len(request.queries)
                )
            )

        # Step 2: Verify requested reader
        try:
            reader = READERS_REGISTRY[request.reader.reader_id]
        except KeyError as err:
            raise Error(
                ErrorMessages.INVALID_READER.value.format(
                    request.reader.reader_id, ", ".join(READERS_REGISTRY.keys())
                )
            ) from err

        # Step 3: Load default reader keyword arguments
        reader_kwargs = {
            k: v.default for k, v in reader.__dataclass_fields__.items() if v.init
        }

        # Step 4: If parameters are provided in request then update keyword arguments used to instantiate reader instance
        if request.reader.parameters:
            for parameter in request.reader.parameters:
                if parameter.parameter_id not in reader_kwargs:
                    raise Error(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "reader", parameter.parameter_id
                        )
                    )

                reader_kwargs[parameter.parameter_id] = parameter.value

        try:
            instance = ReaderFactory.get(reader, reader_kwargs)
        except (ValueError, TypeError) as err:
            raise Error(err.args[0]) from err

        # Step 5: Run apply method
        instance_fields = [
            k
            for k, v in instance.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]
        answers_response = []
        try:
            for idx, query in enumerate(request.queries):
                # Step 5.a: Run "apply" per query
                logging.info(
                    "Applying '%s' reader with parameters = %s for query = '%s' and contexts = %s",
                    instance.__class__.__name__,
                    {
                        k: getattr(instance, k) if k in instance_fields else v
                        for k, v in reader_kwargs.items()
                    },
                    query,
                    request.contexts[idx],
                )
                try:
                    # Step 5.a.i: Adjust "predict" request's arguments based on reader type
                    if isinstance(instance, ExtractiveReader):
                        predictions = instance.predict(
                            questions=[query] * len(request.contexts[idx]),
                            contexts=[[text] for text in request.contexts[idx]],
                            **reader_kwargs,
                        )

                    elif isinstance(instance, GenerativeFiDReader):
                        predictions = instance.predict(
                            questions=[query],
                            contexts=[request.contexts[idx]],
                            **reader_kwargs,
                        )

                    else:
                        raise Error(
                            ErrorMessages.INVALID_READER.value.format(
                                request.reader.reader_id,
                                ", ".join(READERS_REGISTRY.keys()),
                            )
                        )

                    logging.info(
                        "Applying '%s' reader for query = '%s' returns predictions = %s",
                        instance.__class__.__name__,
                        query,
                        predictions.values(),
                    )

                    # Step 5.b: Add answers for current query into response object
                    # `predictions` is a dictionary with <question_id, list of answers per context>
                    for predictions_for_context in predictions.values():
                        answers_per_context = []

                        # Iterate over predictions for current query to formulate answer response object
                        for prediction in predictions_for_context:
                            # Step 5.b.i: Populate mandatory fields
                            answer = {
                                "text": prediction["span_answer_text"],
                                "confidence_score": prediction["confidence_score"],
                            }
                            # Step 5.b.ii: Populate optional fields
                            if (
                                "passage_index" in prediction
                                and prediction["passage_index"]
                            ):
                                answer["context_index"] = int(
                                    prediction["passage_index"]
                                )

                            if (
                                "span_answer" in prediction
                                and prediction["span_answer"]
                            ):
                                answer["start_char_offset"] = prediction["span_answer"][
                                    "start_position"
                                ]
                                answer["end_char_offset"] = prediction["span_answer"][
                                    "end_position"
                                ]

                            # Step 5.b.iii: Add answer to answers_for_question
                            answers_per_context.append(answer)

                        answers_response.append(answers_per_context)

                except TypeError as err:
                    raise Error(
                        ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                            f"{request.reader.reader_id} reader"
                        )
                    ) from err

        except IndexError as err:
            raise Error(
                ErrorMessages.MISSING_CONTEXT.value.format(
                    len(request.contexts), len(request.queries)
                )
            ) from err
        # Step 5: Return
        return answers_response

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
