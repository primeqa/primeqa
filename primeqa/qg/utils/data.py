import logging
import re
from collections import defaultdict
from typing import Dict

import numpy
import torch
from datasets import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def dicts_to_feature_dict(dict_iter):
    output = defaultdict(list)
    for _dict in dict_iter:
        for key, value in _dict.items():
            output[key].append(value)
    return output


def prepare_labelled_data(
    data: Dataset,
    answer_column: str = "answers",
    question_column: str = "question",
    answer_column_move_to: str = "original_answers",
    question_colum_move_to: str = "original_question",
):
    def move_answers_questions(sample: Dict):
        if answer_column is not None and answer_column in sample:
            sample[answer_column_move_to] = sample[answer_column]
            del sample[answer_column]
        if question_column is not None and question_column in sample:
            sample[question_colum_move_to] = sample[question_column]
            del sample[question_column]
        return sample

    return data.map(move_answers_questions)


def select_unique(data: Dataset, column: str, seed=None, verbose: bool = False):
    if seed is not None:
        data = data.shuffle(seed=seed)
    unique_set = set()

    def filter_unique(sample):
        if sample[column] not in unique_set:
            unique_set.add(sample[column])
            return True
        if verbose:
            print(f"Value {sample[column]} appeared multiple times for column {column}")
        return False

    return data.filter(filter_unique, num_proc=1)


def unpack_samples(packed_samples, filter_fn=None):
    samples = {
        "id": [],
        "original_id": [],
        "context": [],
        "question": [],
        "answers": [],
        "score": [],
    }
    for gen_samples in tqdm(packed_samples, desc="Unpacking samples", unit="samples"):
        # gen_samples is a dict
        # apply filter function
        if filter_fn is not None:
            questions, answers, scores = filter_fn(
                context=gen_samples["context"],
                questions=gen_samples["questions"],
                answers=gen_samples["answers"],
                scores=gen_samples["scores"],
            )
        else:
            questions, answers, scores = (
                gen_samples["questions"],
                gen_samples["answers"],
                gen_samples["scores"],
            )
        for idx, (question, answer, score) in enumerate(
            zip(questions, answers, scores)
        ):
            # we append the counter to the original id to use as id for the new sample in order to have unique ids
            samples["id"].append(f"{gen_samples['id']}_{idx}")
            samples["original_id"].append(gen_samples["id"])
            samples["context"].append(gen_samples["context"])
            samples["question"].append(question)
            samples["answers"].append(answer)
            samples["score"].append(score)
    return Dataset.from_dict(samples)


def get_per_sample_indices(separator, *to_split):
    # NOTE this does expect that chunks of the same question only occur consecutively
    _, indices, counts = numpy.unique(separator, return_index=True, return_counts=True)
    counts = [counts[index] for index in numpy.argsort(indices)]
    split_indices = torch.arange(len(separator)).split(counts, dim=0)  # torch version
    # split_indices = numpy.split(range(len(separator)), numpy.sort(indices)[1:]) # numpy version
    return (
        [[item[i] for i in indices] for indices in split_indices] for item in to_split
    )  # does the same as torch.split but for any datatype
    return (torch.tensor(item).split(counts, dim=0) for item in to_split)


def find_answer_span(context, answer, start_char: int = None):
    matched_spans = [
        (match.start(), match.end() - 1)
        for match in re.finditer(re.escape(answer), context)
    ]
    if start_char is not None:
        closest_span_idx = None
        best_diff = None
        for idx, (matched_span_start, _) in enumerate(matched_spans):
            if best_diff is None or abs(matched_span_start - start_char) < best_diff:
                best_diff = abs(matched_span_start - start_char)
                closest_span_idx = idx

        if closest_span_idx is None:
            # answer not found within context
            return -1, -1
        span_start_char, span_end_char = matched_spans[closest_span_idx]
    else:
        span_start_char, span_end_char = matched_spans[0]
    extraced_answer = context[span_start_char : span_end_char + 1]
    assert answer == extraced_answer
    if start_char is not None:
        logger.debug(
            f"Corrected answer span '{answer}': previous span was '{context[start_char:start_char + len(answer)]}' {start_char, start_char + len(answer) - 1} and new span is '{extraced_answer}' {span_start_char, span_end_char}"
        )
    return span_start_char, span_end_char
