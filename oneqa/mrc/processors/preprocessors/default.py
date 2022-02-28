import enum
import itertools
import uuid
from operator import sub
from typing import Optional

from transformers import PreTrainedTokenizerFast
from datasets import Dataset

from oneqa.mrc.processors.preprocessors.abstract import AbstractPreProcessor


class DefaultPreProcessor(AbstractPreProcessor):  # todo better name?
    _yes_no_dict = {'NONE': -1, 'NO': 0, 'YES': 1}

    def adapt_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def process_train(self, examples):
        return self._process(examples, is_train=True)

    def process_eval(self, examples):
        return self._process(examples, is_train=False)

    def _process(self, examples, is_train):
        examples_question = examples['question']
        examples_context = examples['context']
        if isinstance(examples_question, str):  # wrap single (question, [context]) pair in list
            examples_question = [examples_question]
            examples_context = [examples_context]
        examples_id = list(examples.get('example_id', (str(uuid.uuid4()) for _ in range(len(examples_question)))))
        target = examples.get('target')
        
        # create 1:1 question:context lists
        expanded_examples_question = []
        expanded_examples_idx = []
        for i, (question, context) in enumerate(zip(examples_question, examples_context)):
            expanded_examples_question.extend(itertools.repeat(question, len(context)))
            expanded_examples_idx.extend(itertools.repeat(i, len(context)))
        expanded_examples_context = list(itertools.chain.from_iterable(examples_context))

        pad_on_right = self._tokenizer.padding_side == "right"

        tokenized_examples = self._tokenizer(
            expanded_examples_context if pad_on_right else expanded_examples_question,
            expanded_examples_question if pad_on_right else expanded_examples_context,
            stride=self._stride,
            max_length=self._max_seq_len,
            truncation='only_first' if pad_on_right else 'only_second',
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        # oidx      [0, 0, 1, 2, 2, 2]
        # eidx      [0, 0, 1, 1, 1, 1]
        # cidx      [0, 0, 0, 1, 1, 1] (sub oidx eidx)
        # widx      [0, 1, 0, 0, 1, 2] (groupby cidx group idx)
        # tokenized_examples['examples_idx'] = expanded_examples_idx
        tokenized_examples['examples_idx'] = [tokenized_examples["overflow_to_sample_mapping"][eidx] for eidx in expanded_examples_idx]
        tokenized_examples['context_idx'] = list(map(sub, tokenized_examples["overflow_to_sample_mapping"], tokenized_examples['examples_idx']))
        tokenized_examples['window_idx'] = [idx for _, group in itertools.groupby(tokenized_examples['context_idx']) for idx, _ in enumerate(group)]
        tokenized_examples['example_id'] = [examples_id[eidx] for eidx in tokenized_examples['examples_idx']]

        if target:
            tokenized_examples = self._create_targets(tokenized_examples, target, pad_on_right, is_train)

        if target:
            for key in ['start_positions', 'end_positions', 'passage_indices', 'yes_no_answer']:  # TODO 'text' (generative support)
                value = target[key]
                if is_train:
                    value=value[:1]
                if key == 'yes_no_answer':
                    value = [self._yes_no_dict[item] for item in value]
                tokenized_examples['target'][key] = value
        
        return tokenized_examples

    def _create_targets(self, tokenized_examples, target, pad_on_right, is_train):
        
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["passage_indices"] = []
        tokenized_examples["yes_no_answer"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self._tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
        return tokenized_examples