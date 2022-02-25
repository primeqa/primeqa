import itertools
import uuid
from collections import defaultdict
from typing import Optional

from transformers import PreTrainedTokenizerFast
from datasets import Dataset

from oneqa.mrc.processors.preprocessors.abstract import AbstractPreProcessor


class DefaultPreProcessor(AbstractPreProcessor):  # todo better name?
    _yes_no_dict = {'NONE': -1, 'NO': 0, 'YES': 1}

    def __init__(self, tokenizer: PreTrainedTokenizerFast, stride: int, max_q_len: int, max_seq_len: Optional[int] = None):
        super().__init__(tokenizer, stride, max_q_len, max_seq_len)

    def adapt_dataset(self, dataset: Dataset) -> Dataset:
        return dataset

    def process_train(self, examples):
        return self._process(examples, is_train=True)

    def process_eval(self, examples):
        self._process(examples, is_train=False)

    def _process(self, examples, is_train):
        examples_question = examples['question']
        examples_context = examples['context']
        if isinstance(examples_question, str):
            examples_question = [examples_question]
            examples_context = [examples_context]
        examples_id = examples.get('example_id', (str(uuid.uuid4()) for _ in range(len(examples_question))))
        target = examples.get('target')
        if target is None:
            target = itertools.repeat(None, len(examples_question))

        tokenized_examples = defaultdict(list)
        tokenized_examples['target'] = {}

        for question, context, example_id, target in zip(examples_question, examples_context, examples_id, target):
            tokenized_contexts = self._tokenizer(
                context,
                stride=self._stride,
                max_length=self._max_c_len,
                return_overflowing_tokens=True,
                add_special_tokens=False,
                return_offsets_mapping=True,
                truncation=True,
            )

            tokenized_questions = self._tokenizer(
                question,
                max_length=self._max_q_len,
                truncation=True,
                add_special_tokens=False,
            )

            feature_names = set(tokenized_questions.keys())

            tokenized_idx = itertools.count()
            for context_idx, context_idxs in itertools.groupby(tokenized_contexts['overflow_to_sample_mapping']):
                for window_idx, _ in enumerate(context_idxs):
                    idx = next(tokenized_idx)

                    tokenized_examples['example_id'].append(example_id)
                    tokenized_examples['context_idx'].append(context_idx)
                    tokenized_examples['window_idx'].append(window_idx)

                    # model_inputs = {}
                    qp_pair = (tokenized_questions['input_ids'], tokenized_contexts['input_ids'][idx])
                    tokenized_examples['input_ids'].append(self._tokenizer.build_inputs_with_special_tokens(*qp_pair))
                    if 'token_type_ids' in feature_names:
                        tokenized_examples['token_type_ids'].append(self._tokenizer.create_token_type_ids_from_sequences(*qp_pair))
                    if 'attention_mask' in feature_names:
                        # special_tokens_mask = self._tokenizer.get_special_tokens_mask(*qp_pair)
                        tokenized_examples['attention_mask'].append([1] * len(tokenized_examples['input_ids'][-1]))

                    if target:
                        # offset_mapping = tokenized_contexts['offset_mapping']['idx']
                        # start_offsets, end_offsets = tuple(zip(*offset_mapping))
                        # targets = [
                        #     Target(
                        #         position=Position(
                        #             start=t['position']['start'],  # TODO adjust positions, map to tokenized space
                        #             end=t['position']['end'],
                        #             passage=t['position']['passage'],
                        #         ),
                        #         text=t['text'],
                        #         y_n_answer=t.get('y_n_answer')
                        #     )
                        #     for t in target
                        # ]
                        # start_position = []
                        # end_position = []
                        # passage_position = []
                        # y_n_answer = []
                        # text = []
                        # for t in target[:1 if is_train else len(target)]:
                        #     start_position.append(t['position']['start']) # TODO adjust positions, map to tokenized space
                        #     end_position.append(t['position']['end'])
                        #     passage_position.append(t['position']['passage'])
                        #     text.append(t['text'])
                        #     y_n_answer.append(t.get(y_n_answer))

                        # tokenized_examples['start_positions'].append(start_position)
                        # tokenized_examples['end_positions'].append(end_position)
                        # tokenized_examples['passage_positions'].append(passage_position)
                        # tokenized_examples['text'].append(text)
                        # tokenized_examples['yes_no_answer'].append(y_n_answer)
                        for key in ['start_positions', 'end_positions', 'passage_indices', 'yes_no_answer']:  # TODO 'text' (generative support)
                            value = target[key]
                            if is_train:
                                value=value[:1]
                            if key == 'yes_no_answer':
                                value = [self._yes_no_dict[item] for item in value]
                            tokenized_examples['target'][key] = value

                    # else:
                    #     targets = None

                    # feature = InputFeatures(
                    #     example_id=example_id,
                    #     context_idx=context_idx,
                    #     window_idx=window_idx,
                    #     model_inputs=model_inputs,
                    #     targets=targets,
                    # )
                    # yield feature  # TODO need to use HF dataset rather than custom class
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
        
        return Dataset.from_dict(tokenized_examples)
        #return tokenized_examples
