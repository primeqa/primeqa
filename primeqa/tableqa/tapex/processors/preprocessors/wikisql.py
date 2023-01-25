import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional
from copy import deepcopy

import numpy as np
import pandas as pd
from primeqa.tableqa.tapex.utils.wikisql_utils_for_tapex import _TYPE_CONVERTER, retrieve_wikisql_query_answer_tapas
from transformers import TapexTokenizer



def preprocess_tableqa_function_wikisql(examples, model_args,data_args,is_training=False):
    """
    The is_training FLAG is used to identify if we could use the supervision
    to truncate the table content if it is required.
    """

    # load tapex tokenizer
    tokenizer = TapexTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
    # this function is specific for WikiSQL since the util function need the data structure
    # to retrieve the WikiSQL answer for each question
    def _convert_table_types(_table):
        """Runs the type converter over the table cells."""
        ret_table = deepcopy(_table)
        types = ret_table["types"]
        ret_table["real_rows"] = ret_table["rows"]
        typed_rows = []
        for row in ret_table["rows"]:
            typed_row = []
            for column, cell_value in enumerate(row):
                typed_row.append(_TYPE_CONVERTER[types[column]](cell_value))
            typed_rows.append(typed_row)
        ret_table["rows"] = typed_rows
        return ret_table

    questions = [question.lower() for question in examples["question"]]
    example_tables = examples["table"]
    example_sqls = examples["sql"]
    tables = [
        pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
        for example_table in example_tables
    ]

    # using tapas utils to obtain wikisql answer
    answers = []
    for example_sql, example_table in zip(example_sqls, example_tables):
        tapas_table = _convert_table_types(example_table)
        answer_list: List[str] = retrieve_wikisql_query_answer_tapas(tapas_table, example_sql)
        # you can choose other delimiters to split each answer
        answers.append(answer_list)

    padding = "max_length" if data_args.pad_to_max_length else False
    # IMPORTANT: we cannot pass by answers during evaluation, answers passed during training are used to
    # truncate large tables in the train set!
    if is_training:
        model_inputs = tokenizer(
            table=tables,
            query=questions,
            answer=answers,
            max_length=data_args.max_source_length,
            padding=padding,
            truncation=True,
        )
    else:
        model_inputs = tokenizer(
            table=tables, query=questions, max_length=data_args.max_source_length, padding=padding, truncation=True
        )

    labels = tokenizer(
        answer=[", ".join(answer) for answer in answers],
        max_length=data_args.max_target_length,
        padding=padding,
        truncation=True,
    )

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length" and data_args.ignore_pad_token_for_loss:
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs