from datasets.arrow_dataset import Batch
from transformers import BatchEncoding
from datasets import Dataset
from typing import Tuple, List
import torch
import random

from primeqa.mrc.processors.preprocessors.abstract import AbstractPreProcessor
from primeqa.mrc.data_models.subsample_type import SubsampleType
from primeqa.mrc.data_models.target_type import TargetType

class OpenNQPreprocessor(AbstractPreProcessor):
    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        pass

    def label_features_for_subsampling(self, tokenized_examples: BatchEncoding, examples: Batch) -> BatchEncoding:
        pass

    def validate_schema(self, dataset: Dataset, is_train: bool, pre_adaptation: bool = True) -> None:
        pass

    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=True)

    def process_eval(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=False)

    def _process(self, examples: Dataset, is_train: bool) -> Tuple[Dataset, Dataset]:
        """
        Provides implementation for public processing methods.
        """
        if examples.num_rows == 0:
            raise ValueError("No examples to process")

        features = examples.map(
            self._process_example,
            fn_kwargs=dict(is_train=is_train),
            batched=True,
            batch_size=1,
            with_indices=True,
            num_proc=self._num_workers,
            remove_columns=examples.column_names,
            load_from_cache_file=self._load_from_cache_file,
            desc=f"Running featurization on {'train' if is_train else 'eval'} dataset",
        )

        if is_train:
            features = self.subsample_features(features)

        return examples, features

    def _process_example(self, example, idx, is_train) -> BatchEncoding:
        assert(len(example['passages'])== 1)
        id = example['id']
        question = example['question']
        answers = example["answers"] if "answers" in example else ""

        texts = [p['text'] for p in example['passages'][0]]
        titles = [p['title'] for p in example['passages'][0]]
        scores = [p['score'] for p in example['passages'][0]]
        normalized_scores = [p['normalized_score'] for p in example['passages'][0]]
        starts = [p['start'] for p in example['passages'][0]]
        ends = [p['end'] for p in example['passages'][0]]
        assert(len(texts) == len(titles))
        assert(len(texts) == len(scores))
        assert(len(texts) == len(normalized_scores))
        assert(len(texts) == len(starts))
        assert(len(texts) == len(ends))
        for i in range(len(starts)):
            assert(starts[i] <= ends[i])

        if isinstance(question, str):
            question = [question]
        questions = question * len(texts)
        questions = [q.lstrip()[:self._max_q_char_len] for q in questions]

        features = self._tokenizer(
            questions if self._pad_on_right else texts,
            texts if self._pad_on_right else questions,
            stride=0,
            max_length=self._max_seq_len,
            padding='max_length',
            truncation='only_second' if self._pad_on_right else 'only_first',
            return_overflowing_tokens=False,
            return_offsets_mapping=True,
        )
        assert(len(features['input_ids']) == len(texts))
        features['example_idx'] = idx * len(features['input_ids'])
        features['example_id'] = [id[0] + '_' + str(p['rank']) for p in example['passages'][0]]

        if is_train:
            features = self._create_train_targets(features, starts, ends)
            features['subsample_type'] = []
            for i in range(len(features['input_ids'])):
                if features['target_type'][i] == TargetType.SPAN_ANSWER:
                    features['subsample_type'].append(SubsampleType.POSITIVE)
                else:
                    features['subsample_type'].append(SubsampleType.NEGATIVE_NO_ANSWER)
        else:
            context_index = 1 if self._pad_on_right else 0
            for i in range(len(features["input_ids"])):
                # Grab the sequence corresponding to that example (to know what is the context and what is the question).
                sequence_ids = features.sequence_ids(i)
                # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
                # position is part of the context or not.
                features["offset_mapping"][i] = [
                    (o if sequence_ids[k] == context_index else None)
                    for k, o in enumerate(features["offset_mapping"][i])
                ]
        return features


    def _create_train_targets(self, features, starts, ends) -> BatchEncoding:
        """
            Create start/end position and target type targets for training.
        """
        features["start_positions"] = []
        features["end_positions"] = []
        features["target_type"] = []
        offset_mapping = features["offset_mapping"]

        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = features["input_ids"][i]
            cls_index = input_ids.index(self._tokenizer.cls_token_id)
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = features.sequence_ids(i)

            start_position = starts[i]
            end_position = ends[i]
            if start_position < 0 or end_position < 0:
                features["start_positions"].append(cls_index)
                features["end_positions"].append(cls_index)
                features["target_type"].append(TargetType.NO_ANSWER)
                continue

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if self._pad_on_right else 0):
                token_start_index += 1
            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if self._pad_on_right else 0):
                token_end_index -= 1

            while token_start_index <= token_end_index and offsets[token_start_index][0] < start_position:
                token_start_index += 1
            if token_start_index > token_end_index:
                features["start_positions"].append(cls_index)
                features["end_positions"].append(cls_index)
                features["target_type"].append(TargetType.NO_ANSWER)
                continue
            while token_end_index >= token_start_index and offsets[token_end_index][1] > end_position:
                token_end_index -= 1
            if token_end_index < token_start_index:
                features["start_positions"].append(cls_index)
                features["end_positions"].append(cls_index)
                features["target_type"].append(TargetType.NO_ANSWER)
                continue

            features["start_positions"].append(token_start_index)
            features["end_positions"].append(token_end_index)
            features["target_type"].append(TargetType.SPAN_ANSWER)

        return features

    def subsample_features(self, dataset: Dataset) -> Dataset:
        if self._negative_sampling_prob_when_has_answer == 1.0 and \
            self._negative_sampling_prob_when_no_answer == 1.0:
            return dataset

        keep_indices = [i for i, st in enumerate(dataset['subsample_type']) if self._keep_feature(st)]
        if len(keep_indices) == 0:
            raise ValueError("No features remaining after subsampling")

        dataset = dataset.select(keep_indices)

        dataset = dataset.remove_columns('subsample_type')
        return dataset

    def _keep_feature(self, st: SubsampleType) -> bool:
        """
        Return True iff this training feature should be kept based on the subsample type.

        Raises:
            NotImplementedError: invalid SubsampleType value.
        """
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
        """
        Returns true iff tokenizer pads on right side.
        """
        return self._tokenizer.padding_side == "right"

