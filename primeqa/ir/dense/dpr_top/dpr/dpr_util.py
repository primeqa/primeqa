from transformers import (DPRQuestionEncoder, RagTokenizer, DPRQuestionEncoderTokenizer, RagTokenForGeneration)
import logging
from typing import List, Union, Tuple
from primeqa.ir.dense.dpr_top.torch_util.hypers_base import HypersBase

logger = logging.getLogger(__name__)


class DPROptions(HypersBase):
    def __init__(self):
        super().__init__()

    def _post_init(self):
        super()._post_init()

def tokenize_queries(tokenizer: Union[RagTokenizer, DPRQuestionEncoderTokenizer],
                     queries: Union[List[str], List[Tuple[str, str]]], *, max_length: int):
    """

    :param tokenizer: a RagTokenizer (the question_encoder will be used) or a DPRQuestionEncoderTokenizer
    :param queries: list of string queries or list of title/text pairs
    :param max_length:
    :return:
    """
    if hasattr(tokenizer, 'question_encoder'):
        # if this is a RagTokenizer, pull out the question encoder tokenizer
        tokenizer = tokenizer.question_encoder
    if max_length is None:
        max_length = tokenizer.model_max_length
    if type(queries[0]) == str:
        assert all(type(q) == str for q in queries)
        # single sequence queries
        model_inputs = tokenizer(
            queries,
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            padding="longest",
            truncation=True,
        )
    else:
        assert all(type(q) == tuple and len(q) == 2 and type(q[0]) == str and type(q[1]) == str for q in queries)
        # title / text queries
        model_inputs = tokenizer(
            [q[0] for q in queries],
            [q[1] for q in queries],
            add_special_tokens=True,
            return_tensors="pt",
            max_length=max_length,
            padding="longest",
            truncation=True,
        )
    return model_inputs


def queries_to_vectors(tokenizer: Union[RagTokenizer, DPRQuestionEncoderTokenizer], question_encoder,
                       queries: Union[List[str], List[Tuple[str, str]]], *, max_query_length=None):
    """

    :param tokenizer: a RagTokenizer (the question_encoder will be used) or a DPRQuestionEncoderTokenizer
    :param question_encoder:
    :param queries: list of string queries or list of title/text pairs
    :param max_query_length:
    :return:
    """
    input_dict = tokenize_queries(tokenizer, queries, max_length=max_query_length)
    input_ids = input_dict['input_ids'].to(question_encoder.device)
    attention_mask = input_dict['attention_mask'].to(question_encoder.device)

    question_enc_outputs = question_encoder(
        input_ids, attention_mask=attention_mask, return_dict=True
    )
    question_encoder_last_hidden_state = question_enc_outputs[0]  # hidden states of question encoder
    return question_encoder_last_hidden_state

