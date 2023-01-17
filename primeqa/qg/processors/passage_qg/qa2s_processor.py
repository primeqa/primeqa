from dataclasses import dataclass
from typing import Dict

import numpy
import torch
from datasets import Dataset
from primeqa.qg.utils.data import get_per_sample_indices
from transformers import PreTrainedTokenizerFast


@dataclass
class QA2SProcessor:
    """
    Class for QA2S processing, contains methods to preprocess data
    """

    tokenizer: PreTrainedTokenizerFast
    input_max_len: int = None
    target_max_len: int = None
    max_context_length: int = None
    max_question_length: int = None
    stride: int = 128
    prefix_question_whitespace: int = True

    def __call__(self, dataset) -> Dataset:
        return dataset.map(
            self.preprocess_data,
            fn_kwargs={
                "tokenizer": self.tokenizer,
                "input_max_length": self.input_max_len,
                "target_max_length": self.target_max_len,
                "max_context_length": self.max_context_length,
                "max_question_length": self.max_question_length,
                "stride": self.stride,
                "prefix_question_whitespace": self.prefix_question_whitespace,
            },
            batched=True,
        )

    @staticmethod
    def preprocess_data(
        example_batch: Dict,
        tokenizer: PreTrainedTokenizerFast,
        target_max_length: int,
        max_context_length: int,
        max_question_length: int,
        input_max_length: int,
        stride: int,
        question_column: str = "question",
        context_column: str = "context",
        answer_column: str = "answers",
        fill_missing_columns: bool = True,
        prefix_question_whitespace: bool = True,
    ):
        question_column = "question" if isinstance(question_column, bool) and question_column else question_column
        answer_column = "answers" if isinstance(answer_column, bool) and answer_column else answer_column

        # allow int values to be -1 (means None)
        if target_max_length == -1:
            target_max_length = None
        if max_context_length == -1:
            max_context_length = None
        if max_question_length == -1:
            max_question_length = None
        if input_max_length == -1:
            input_max_length = None

        soc_token_id = tokenizer.encode("<s>", add_special_tokens=False)
        soa_token_id = tokenizer.encode("<a>", add_special_tokens=False)
        soq_token_id = tokenizer.encode("<q>", add_special_tokens=False)
        assert len(soc_token_id) == len(soa_token_id) == len(soq_token_id) == 1
        soc_token_id = soc_token_id[0]
        soa_token_id = soa_token_id[0]
        soq_token_id = soq_token_id[0]

        if max_context_length is not None:
            # Truncate contexts (from behind) to max length
            # It's ok to tokenize string and convert back since the same tokenizer (uncased or cased) is used for tokenization later on anyway
            example_batch[context_column] = [
                tokenizer.convert_tokens_to_string(tokenizer.tokenize(context)[-max_context_length:])
                for context in example_batch[context_column]
            ]
        if question_column in example_batch and max_question_length is not None:
            # Truncate questions (from behind) to max length
            # It's ok to tokenize string and convert back since the same tokenizer (uncased or cased) is used for tokenization later on anyway
            example_batch[question_column] = [
                tokenizer.convert_tokens_to_string(tokenizer.tokenize(question)[-max_question_length:])
                for question in example_batch[question_column]
            ]

        if question_column in example_batch and answer_column in example_batch:
            # we need questions and answers for labels
            sequence_1 = example_batch[context_column]
            # first part of qa2s
            last_seq_gen_start_id = soq_token_id
            targets = [
                "<q>" + (" " if prefix_question_whitespace else "") + question + "</q>"
                for question in example_batch[question_column]
            ]

            if input_max_length is None:
                input_max_length = tokenizer.model_max_length
            input_max_length = (
                input_max_length - 1
            )  # -1 accounts for the sos token at the beginning of the sequence which is added after tokenization (to ensure every chunk starts with it)
        else:
            # inference samples
            sequence_1 = example_batch[context_column]
            # first part of qa2s
            # NOTE: for inference we can only prepare inputs for question generation in case of qa2s
            seq_after_context_start_id = soq_token_id
            last_seq_gen_start_id = soq_token_id

            if input_max_length is None:
                input_max_length = tokenizer.model_max_length
            input_max_length = (
                input_max_length - 1
            )  # -1 accounts for the CLS token at the beginning of the sequence which is added later (to ensure every chunk starts with it)

        max_length_2 = input_max_length
        if max_question_length is not None:
            # for `qa2s` we might want to set a maximum question length for the first step so that the input for the second decoding step fits into the model after generating the question (and concatenating it to the input)
            input_max_length -= max_question_length

        tokenized_samples = tokenizer(
            sequence_1,
            truncation="only_first",
            stride=stride,
            max_length=input_max_length,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=False,
            return_length=True,
            padding=False,
            add_special_tokens=False,
        )
        # add cls token in front of every chunk
        tokenized_samples["input_ids"] = [[soc_token_id] + input_ids for input_ids in tokenized_samples["input_ids"]]
        tokenized_samples["attention_mask"] = [
            [1] + attention_mask for attention_mask in tokenized_samples["attention_mask"]
        ]
        tokenized_samples["length"] = [length + 1 for length in tokenized_samples["length"]]

        # compute attention mask without special tokens for sentence embeddings
        indices = [
            numpy.where(numpy.array(tokenized_samples.sequence_ids(i)) == None)[0].tolist()
            for i in range(len(tokenized_samples["input_ids"]))
        ]
        tokenized_samples["attention_mask_without_special_tokens"] = [
            torch.tensor(attention_mask).index_fill(0, torch.LongTensor([0] + special_tokens_indices), 0).tolist()
            for special_tokens_indices, attention_mask in zip(indices, tokenized_samples["attention_mask"])
        ]

        # print([len(_id) for _id in tokenized_samples['input_ids']])
        # print(max(len(_id) for _id in tokenized_samples['input_ids']))
        # exit()

        if question_column in example_batch:
            # set up tokenizer for targets in case of sequence-to-sequence models
            with tokenizer.as_target_tokenizer():
                tokenized_labels = tokenizer(
                    targets,
                    padding=False,
                    truncation=True,
                    return_overflowing_tokens=False,
                    add_special_tokens=False,
                    max_length=target_max_length,
                )
                # shift so that tokens < n predict n
                decoder_input_ids = [input_ids[:-1] for input_ids in tokenized_labels["input_ids"]]
                decoder_labels = [input_ids[1:] for input_ids in tokenized_labels["input_ids"]]

        # we skip the chunks where the answer is not within the context therefore we store our positie samples in a separate dict
        processed_samples = {}
        processed_samples["input_ids"] = []
        processed_samples["attention_mask"] = []
        processed_samples["attention_mask_without_special_tokens"] = []

        if question_column in example_batch and answer_column in example_batch:
            processed_samples["labels"] = []
            processed_samples["question"] = []
            processed_samples["answers"] = []
            processed_samples["has_answer"] = []
            # separate decoder input ids with bos token in the beginning (and missong eos token in the end)
            processed_samples["decoder_input_ids"] = []

        # we mark samples so that we know for which step they are
        processed_samples["qa2s_step"] = []

        length = tokenized_samples.pop("length")
        overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
        (per_sample_chunk_indices,) = get_per_sample_indices(
            overflow_to_sample_mapping, range(len(overflow_to_sample_mapping))
        )

        if fill_missing_columns:
            remaining_keys = example_batch.keys() - processed_samples.keys()
            for _key in remaining_keys:
                processed_samples[_key] = []

        for chunk_indices in per_sample_chunk_indices:
            for i in chunk_indices:
                # sample index before tokenizing to retrieve correct information (id, context etc.)
                sample_idx = overflow_to_sample_mapping[i]

                offset_mapping = tokenized_samples["offset_mapping"][i]

                context_end_index = length[i] - 1

                if question_column in example_batch and answer_column in example_batch:
                    # check if answer is within current chunk so that we can mark this sample using has_answer for later filtering
                    answers = example_batch[answer_column][sample_idx]
                    has_answer = False
                    for start_char, text in zip(answers["answer_start"], answers["text"]):
                        end_char = start_char + len(text) - 1

                        # make sure that index is correct
                        # some datasets have wrong cases hence we normalize the answers (which is done anyway in the evaluation of the model's predictions) since rectifying answers wouldn't solve the issue as it is case-sensitive
                        assert (example_batch[context_column][sample_idx][start_char : end_char + 1]) == (
                            text
                        ), f"Char span is wrong, make sure to run data correction first. Extracted answer is '{example_batch[context_column][sample_idx][start_char:end_char + 1]}' but given answer is '{text}'"

                        # make sure that end_char is not beyond last context char (there might be index erorrs in the annotated data)
                        end_char = min(end_char, len(example_batch["context"][sample_idx]) - 1)

                        # determine whether answer is within the current chunk
                        start_token = 0
                        # we have to start at the last token of the context since the first sep_token belongs already to the second sequence hence having offsets starting at 0 again
                        # moreover offset_mapping starts at input_ids index 1 (as well as sequence_ids)
                        end_token = context_end_index - 1
                        if offset_mapping[start_token][0] <= start_char and offset_mapping[end_token][1] > end_char:
                            # answer is within current chunk
                            has_answer = True

                            # find start token by looping over context until char mapping is >= start_char
                            while (
                                offset_mapping[start_token][1] - 1 < start_char
                            ):  # additional constraint here since start char might fall into last context token
                                start_token += 1

                            # find end token likewise
                            while offset_mapping[end_token][0] > end_char:
                                end_token -= 1

                            assert start_token <= end_token

                            # we can leave the loop here since having one answer within the current chunk is enough
                            break
                    processed_samples["has_answer"].append(has_answer)

                if fill_missing_columns:
                    # add missing columns
                    for _key in remaining_keys:
                        processed_samples[_key].append(example_batch[_key][sample_idx])

                # add sample properties to output (since we do not consider all chunks)
                processed_samples["input_ids"].append(tokenized_samples["input_ids"][i])
                processed_samples["attention_mask"].append(tokenized_samples["attention_mask"][i])
                processed_samples["attention_mask_without_special_tokens"].append(
                    tokenized_samples["attention_mask_without_special_tokens"][i]
                )

                # mark instance belonging to first step
                processed_samples["qa2s_step"].append(0)

                if question_column in example_batch and answer_column in example_batch:
                    processed_samples["question"].append(example_batch[question_column][sample_idx])
                    processed_samples["answers"].append(example_batch[answer_column][sample_idx])

                    # for seq2seq models we don't have to mask any labels
                    processed_samples["decoder_input_ids"].append(decoder_input_ids[sample_idx])
                    labels = decoder_labels[sample_idx]
                    processed_samples["labels"].append(labels)

        if question_column in example_batch and answer_column in example_batch:
            # second step of qa2s if labels are given (otherwise input has to be prepared after question has been generated)
            sequence_1 = example_batch[context_column]
            targets = [
                "<a>" + ("" if answer["answer_start"][0] == 0 else " ") + answer["text"][0] + "</a>"
                for answer in example_batch[answer_column]
            ]
            sequence_2 = [
                "<q>" + (" " if prefix_question_whitespace else "") + question
                for question in example_batch[question_column]
            ]

            # max_length is still set

            tokenized_samples = tokenizer(
                sequence_1,
                sequence_2,
                truncation="only_first",
                stride=stride,
                max_length=max_length_2,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=False,
                return_length=True,
                padding=False,
                add_special_tokens=False,
            )
            # add cls token in front of every chunk
            tokenized_samples["input_ids"] = [
                [soc_token_id] + input_ids for input_ids in tokenized_samples["input_ids"]
            ]
            tokenized_samples["attention_mask"] = [
                [1] + attention_mask for attention_mask in tokenized_samples["attention_mask"]
            ]
            tokenized_samples["length"] = [length + 1 for length in tokenized_samples["length"]]

            # compute attention mask without special tokens for sentence embeddings
            indices = [
                numpy.where(numpy.array(tokenized_samples.sequence_ids(i)) == None)[0].tolist()
                for i in range(len(tokenized_samples["input_ids"]))
            ]
            tokenized_samples["attention_mask_without_special_tokens"] = [
                torch.tensor(attention_mask).index_fill(0, torch.LongTensor([0] + special_tokens_indices), 0).tolist()
                for special_tokens_indices, attention_mask in zip(indices, tokenized_samples["attention_mask"])
            ]

            # set up tokenizer for targets in case of sequence-to-sequence models
            with tokenizer.as_target_tokenizer():
                tokenized_labels = tokenizer(
                    targets,
                    padding=False,
                    truncation=True,
                    add_special_tokens=False,
                    max_length=target_max_length,
                )
                # shift so that tokens < n predict n
                decoder_input_ids = [input_ids[:-1] for input_ids in tokenized_labels["input_ids"]]
            decoder_labels = [input_ids[1:] for input_ids in tokenized_labels["input_ids"]]

            length = tokenized_samples.pop("length")
            overflow_to_sample_mapping = tokenized_samples.pop("overflow_to_sample_mapping")
            (per_sample_chunk_indices,) = get_per_sample_indices(
                overflow_to_sample_mapping, range(len(overflow_to_sample_mapping))
            )

            for chunk_indices in per_sample_chunk_indices:
                for i in chunk_indices:
                    # sample index before tokenizing to retrieve correct information (id, context etc.)
                    sample_idx = overflow_to_sample_mapping[i]

                    offset_mapping = tokenized_samples["offset_mapping"][i]

                    # for non-seq2seq and config `qa2s` (2nd step with question in input) we need to figure out last token before question
                    input_ids_tensor = torch.tensor(tokenized_samples["input_ids"][i])
                    context_end_index = (input_ids_tensor == soq_token_id).nonzero(as_tuple=True)[0][0].item() - 1

                    # check if answer is within current chunk so that we can mark this sample using has_answer for later filtering
                    answers = example_batch[answer_column][sample_idx]
                    has_answer = False
                    for start_char, text in zip(answers["answer_start"], answers["text"]):
                        end_char = start_char + len(text) - 1

                        # make sure that index is correct
                        # some datasets have wrong cases hence we normalize the answers (which is done anyway in the evaluation of the model's predictions) since rectifying answers wouldn't solve the issue as it is case-sensitive
                        assert (example_batch[context_column][sample_idx][start_char : end_char + 1]) == (
                            text
                        ), f"Char span is wrong, make sure to run data correction first. Extracted answer is '{example_batch[context_column][sample_idx][start_char:end_char + 1]}' but given answer is '{text}'"

                        # make sure that end_char is not beyond last context char (there might be index erorrs in the annotated data)
                        end_char = min(end_char, len(example_batch["context"][sample_idx]) - 1)

                        # determine whether answer is within the current chunk
                        start_token = 0
                        # we have to start at the last token of the context since the first sep_token belongs already to the second sequence hence having offsets starting at 0 again
                        # moreover offset_mapping starts at input_ids index 1 (as well as sequence_ids)
                        end_token = context_end_index - 1
                        if offset_mapping[start_token][0] <= start_char and offset_mapping[end_token][1] > end_char:
                            # answer is within current chunk
                            has_answer = True

                            # find start token by looping over context until char mapping is >= start_char
                            while (
                                offset_mapping[start_token][1] - 1 < start_char
                            ):  # additional constraint here since start char might fall into last context token
                                start_token += 1

                            # find end token likewise
                            while offset_mapping[end_token][0] > end_char:
                                end_token -= 1

                            assert start_token <= end_token

                            # we can leave the loop here since having one answer within the current chunk is enough
                            break
                    processed_samples["has_answer"].append(has_answer)

                    if fill_missing_columns:
                        # add missing columns
                        for _key in remaining_keys:
                            processed_samples[_key].append(example_batch[_key][sample_idx])

                    # add sample properties to output (since we do not consider all chunks)
                    processed_samples["input_ids"].append(tokenized_samples["input_ids"][i])
                    processed_samples["attention_mask"].append(tokenized_samples["attention_mask"][i])
                    processed_samples["attention_mask_without_special_tokens"].append(
                        tokenized_samples["attention_mask_without_special_tokens"][i]
                    )

                    # mark instance belonging to first step
                    processed_samples["qa2s_step"].append(1)

                    processed_samples["question"].append(example_batch[question_column][sample_idx])
                    processed_samples["answers"].append(example_batch[answer_column][sample_idx])

                    processed_samples["decoder_input_ids"].append(decoder_input_ids[sample_idx])
                    labels = decoder_labels[sample_idx]
                    processed_samples["labels"].append(labels)

        return processed_samples
