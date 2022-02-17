import itertools
import uuid
from itertools import chain

import datasets
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from typing import Optional

from oneqa.mrc.processors.features import InputFeatures, Target, Position


class DefaultProcessor:
    def __init__(self, tokenizer: PreTrainedTokenizerFast, stride: int, max_q_len: int, max_seq_len: Optional[int] = None):
        self._tokenizer = tokenizer  # TODO: fast toks only??
        self._stride = stride
        self._max_q_len = max_q_len
        self._max_seq_len = max_seq_len or self._tokenizer.max_len_sentences_pair
        self._max_c_len = self._max_seq_len - self._max_q_len

    def process_train(self, examples):
        pass

    def process_eval(self, examples):
        pass

    def _process(self, examples):
        examples_question = examples['question']
        examples_context = examples['context']
        if isinstance(examples_question, str):
            examples_question = [examples_question]
            examples_context = [examples_context]
        examples_id = examples.get('example_id', [uuid.uuid4() for _ in range(len(examples_question))])
        target = examples.get('target')
        if target is None:
            target = itertools.repeat(None, len(examples_question))

        for question, context, example_id, target in zip(examples_question, examples_context, examples_id, target):
            tokenized_contexts = self._tokenizer(
                context,
                stride=self._stride,
                max_length=self._max_c_len,
                return_overflowing_tokens=True,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )

            tokenized_questions = self._tokenizer(
                question,
                max_length=self._max_q_len,
                truncation=True,
                add_special_tokens=False
            )

            feature_names = set(tokenized_questions.keys())

            tokenized_idx = itertools.count()
            for context_idx, context_idxs in itertools.groupby(tokenized_contexts['overflow_to_sample_mapping']):
                for window_idx, _ in enumerate(context_idxs):
                    idx = next(tokenized_idx)
                    model_inputs = {}
                    qp_pair = (tokenized_questions['input_ids'], tokenized_contexts['input_ids'][idx])
                    model_inputs['input_ids'] = self._tokenizer.build_inputs_with_special_tokens(*qp_pair)
                    if 'token_type_ids' in feature_names:
                        model_inputs['token_type_ids'] = self._tokenizer.create_token_type_ids_from_sequences(*qp_pair)
                    if 'attention_mask' in feature_names:
                        # special_tokens_mask = self._tokenizer.get_special_tokens_mask(*qp_pair)
                        model_inputs['attention_mask'] = [1] * len(model_inputs['input_ids'])

                    if target:
                        offset_mapping = tokenized_contexts['offset_mapping']['idx']
                        start_offsets, end_offsets = tuple(zip(*offset_mapping))
                        targets = [
                            Target(
                                position=Position(
                                    start=t['position']['start'],  # TODO adjust positions, map to tokenized space
                                    end=t['position']['end'],
                                    passage=t['position']['passage'],
                                ),
                                text=t['text'],
                                y_n_answer=t.get('y_n_answer')
                            )
                            for t in target
                        ]
                    else:
                        targets = None

                    feature = InputFeatures(
                        example_id=example_id,
                        context_idx=context_idx,
                        window_idx=window_idx,
                        model_inputs=model_inputs,
                        targets=targets,
                    )
                    yield feature  # TODO need to use HF dataset rather than custom class
            # q_toks = self._tokenizer.tokenize(question)[:self._max_q_len]
            #
            # for context_idx, c in enumerate(context):
            #     ct = self._tokenizer.tokenize(c)[:self._max_c_len]
            #     tokenized = self._tokenizer(
            #         [q_toks],
            #         [ct],
            #         max_length=self._max_seq_len,
            #         stride=self._stride,
            #         return_overflowing_tokens=True,
            #         is_split_into_words=True
            #     )
            #
            #     n_windows = len(tokenized['input_ids'])
            #     for window_idx in range(n_windows):
            #         yield InputFeatures(
            #             example_id=example_id,
            #             context_idx=context_idx,
            #             window_idx=window_idx,
            #             model_inputs={
            #                 feature_name: feature[window_idx]
            #                 for feature_name, feature in tokenized.items()
            #             }
            #         )

