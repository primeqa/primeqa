import enum
import itertools
import random
import uuid
from operator import sub
from typing import Optional, List, Iterable, Tuple, Any, Dict

from datasets.arrow_dataset import Batch
from transformers import BatchEncoding
from datasets import Dataset

from oneqa.mrc.processors.preprocessors.abstract import AbstractPreProcessor
from oneqa.mrc.types.subsample_type import SubsampleType
from oneqa.mrc.types.target_type import TargetType


# TODO type signatures for all methods
class DefaultPreProcessor(AbstractPreProcessor):  # todo better name?
    _del_keys = ["overflow_to_sample_mapping", "offset_mapping"]  #, "context_idx", "window_idx", "example_idx"]  # TODO: remove window_idx if keeping that removed

    def adapt_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=True)

    def process_eval(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=False)

    def _process(self, examples: Dataset, is_train: bool) -> Tuple[Dataset, Dataset]:
        examples = self.adapt_dataset(examples)
        features = examples.map(
            self._process_batch,
            fn_kwargs=dict(is_train=is_train),
            batched=True,
            num_proc=self._num_workers,
            remove_columns=examples.column_names,
            load_from_cache_file=self._load_from_cache_file,
            desc=f"Running tokenizer on {'train' if is_train else 'eval'} dataset",
        )
        if is_train:
            features = self.subsample_features(features)
        return examples, features

    def _process_batch(self, examples: Batch, is_train: bool) -> BatchEncoding:
        examples_question = examples['question']
        examples_context = examples['context']
        if isinstance(examples_question, str):  # wrap single (question, [context]) pair in list
            examples_question = [examples_question]
            examples_context = [examples_context]
        
        # create 1:1 question:context lists
        expanded_examples_question = []
        expanded_examples_idx = []
        for i, (question, context) in enumerate(zip(examples_question, examples_context)):
            expanded_examples_question.extend(itertools.repeat(question, len(context)))
            expanded_examples_idx.extend(itertools.repeat(i, len(context)))
        expanded_examples_context = list(itertools.chain.from_iterable(examples_context))

        tokenized_examples = self._tokenizer(
            expanded_examples_context if self._pad_on_right else expanded_examples_question,
            expanded_examples_question if self._pad_on_right else expanded_examples_context,
            stride=self._stride,
            max_length=self._max_seq_len,
            truncation='only_first' if self._pad_on_right else 'only_second',
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
        )

        examples_id = list(examples.get('example_id', (str(uuid.uuid4()) for _ in range(len(examples_question)))))
        tokenized_examples['example_idx'] = [expanded_examples_idx[oidx] for oidx in tokenized_examples["overflow_to_sample_mapping"]]
        spans_per_example = self._generate_previous_spans_per_example(tokenized_examples['example_idx'], tokenized_examples["overflow_to_sample_mapping"])
        tokenized_examples['context_idx'] = list(map(sub, tokenized_examples["overflow_to_sample_mapping"], spans_per_example))
        # tokenized_examples['window_idx'] = [idx for _, group in itertools.groupby(tokenized_examples['context_idx']) for idx, _ in enumerate(group)]  # TODO: is this needed?
        tokenized_examples['example_id'] = [examples_id[eidx] for eidx in tokenized_examples['example_idx']]

        if is_train:
            tokenized_examples = self._create_train_targets(tokenized_examples, examples['target'])
        else:
            tokenized_examples = self._create_eval_targets(tokenized_examples)

        for key in self._del_keys:
            tokenized_examples.pop(key, None)

        if is_train:
            tokenized_examples = self.label_features_for_subsampling(tokenized_examples, examples)
        
        return tokenized_examples

    def _create_train_targets(self, tokenized_examples: BatchEncoding, target: List[Dict[str, Any]]) -> BatchEncoding:  # TODO: rename create targets fns?
        
        # Since one context might give us several features if it has a long context, and each example can have many contexts,
        # we need a map from a feature to ts corresponding example. This key gives us just that.
        example_mapping = tokenized_examples['example_idx']
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        tokenized_examples["target_type"] = []

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self._tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            example_index = example_mapping[i]
            context_idx = tokenized_examples['context_idx'][i]
            t = target[example_index]
            passage_index = t['passage_indices'][0]
            start_position = t['start_positions'][0]
            end_position = t['end_positions'][0]
            yes_no_answer = TargetType.from_bool_label(t['yes_no_answer'])

            if passage_index == context_idx and start_position == -1:  # Passage Answer
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tt = yes_no_answer
                if tt not in (TargetType.YES, TargetType.NO):
                    tt = TargetType.PASSAGE_ANSWER
                tokenized_examples["target_type"].append(tt)
            elif passage_index != context_idx or start_position == -1:  # No Answer
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
                tokenized_examples["target_type"].append(TargetType.NO_ANSWER)
            else:  # Span Answer
                # Start/end character index of the answer in the text.
                start_char = start_position
                end_char = end_position

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self._pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self._pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                    tokenized_examples["target_type"].append(TargetType.NO_ANSWER)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
                    tokenized_examples["target_type"].append(TargetType.SPAN_ANSWER)

        return tokenized_examples

    def _create_eval_targets(self, tokenized_examples: BatchEncoding) -> BatchEncoding:
        
        context_index = 1 if self._pad_on_right else 0
        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples
    
    def label_features_for_subsampling(self, tokenized_examples: BatchEncoding, examples: Batch) -> BatchEncoding:
        if self._subsample_all_features:
            self._logger.warning("Keeping all negative training instances -- "
                                 "this may create an unbalanced training set and increase training time significantly")
            return tokenized_examples
        elif self._subsample_no_features:
            self._logger.warning("Removing all negative training instances -- only positives will be used")

        example_mapping = tokenized_examples['example_idx']

        tokenized_examples['subsample_type'] = []
        for i in range(len(tokenized_examples['input_ids'])):
            if tokenized_examples['target_type'][i] != TargetType.NO_ANSWER:
                st = SubsampleType.POSITIVE
            else:
                example_idx = example_mapping[i]
                passage_mapping = examples['target'][example_idx]['passage_indices']
                passage_idx = passage_mapping[0]
                has_answer = passage_idx != -1
                if has_answer:
                    st = SubsampleType.NEGATIVE_HAS_ANSWER
                else:
                    st = SubsampleType.NEGATIVE_NO_ANSWER
            tokenized_examples['subsample_type'].append(st)

        return tokenized_examples

    def subsample_features(self, dataset: Dataset) -> Dataset:
        if self._subsample_all_features:
            return dataset

        keep_indices = (i for i, st in enumerate(dataset['subsample_type']) if self._keep_feature(st))
        dataset = dataset.select(keep_indices)
        dataset = dataset.remove_columns('subsample_type')
        return dataset

    def _keep_feature(self, st: SubsampleType) -> bool:
        if st == SubsampleType.POSITIVE:
            return True
        elif st == SubsampleType.NEGATIVE_HAS_ANSWER:
            return random.random() < self._negative_sampling_prob_when_has_answer
        elif st == SubsampleType.NEGATIVE_NO_ANSWER:
            return random.random() < self._negative_sampling_prob_when_no_answer
        else:
            raise NotImplementedError(f"Unexpected subsample type: {st}")

    @property
    def _pad_on_right(self) -> bool:
        return self._tokenizer.padding_side == "right"

    @property
    def _subsample_all_features(self) -> bool:
        return self._negative_sampling_prob_when_has_answer == self._negative_sampling_prob_when_no_answer == 1.

    @property
    def _subsample_no_features(self) -> bool:
        return self._negative_sampling_prob_when_has_answer == self._negative_sampling_prob_when_no_answer == 0.

    @staticmethod
    def _generate_previous_spans_per_example(example_idx: List[int], sample_mapping: List[int]) -> Iterable[int]:
        group_start_idx = 0
        for _, group in itertools.groupby(example_idx):
            for group_len, _ in enumerate(group, 1):
                pass
            yield from itertools.repeat(sample_mapping[group_start_idx], group_len)
            group_start_idx += group_len
