import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import numpy as np
import pandas as pd
from transformers import TapexTokenizer



def preprocess_tableqa_function_wtq(examples, model_args, data_args, is_training=False):
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

    questions = [question.lower() for question in examples["question"]]
    example_tables = examples["table"]
    tables = [
        pd.DataFrame.from_records(example_table["rows"], columns=example_table["header"])
        for example_table in example_tables
    ]

    # using wikitablequestion's answer set
    answers = examples["answers"]
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