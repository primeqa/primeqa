# coding=utf-8
# Copyright 2020 The Google Research Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility function for TyDi QA evaluation."""

import collections

# import gzip
# import glob
# import json
# import logging
# import multiprocessing


# A data structure for storing prediction and annotation.
# When a example has multiple annotations, multiple TyDiLabel will be used.
TyDiLabel = collections.namedtuple(
    'TyDiLabel',
    [
        'plaintext',  # context.
        'question_text',  # a question text.
        'example_id',  # the unique id for each TyDi example.
        'language',  # language id.
        'passage_answer_index',  # A index for passage answer among candidates.
        'passage_span',  # offsets for the passage span
        'minimal_answer_span',  # A Span object for minimal answer.
        'yes_no_answer',  # Indicate if the minimal answer is an yes/no answer
        #   The possible values are "yes", "no", "none".
        #   (case insensitive)
        #   If the field is "yes", minimal_answer_span should be empty or null.
        'passage_score',  # The score for the passage answer prediction.
        'minimal_score'  # The score for the minimal answer prediction.
    ])


class Span(object):
    """A class for handling token and byte spans.

      The logic is:

      1) if both start_byte != -1 and end_byte != -1 then the span is defined
         by byte offsets
      3) else, this is a null span.

      Null spans means that there is no (passage or minimal) answers.

    """

    def __init__(self, start_byte_offset, end_byte_offset):

        if ((start_byte_offset < 0 and end_byte_offset >= 0) or
                (start_byte_offset >= 0 and end_byte_offset < 0)):
            raise ValueError('Inconsistent Null Spans (Byte).')

        if (start_byte_offset >= 0 and end_byte_offset >= 0 and
                start_byte_offset > end_byte_offset):
            raise ValueError('Invalid byte spans (start_byte >= end_byte).')

        self.start_byte_offset = start_byte_offset
        self.end_byte_offset = end_byte_offset

    def is_null_span(self):
        """A span is a null span if the start and end are both -1.

        This can happen for both gold and predicted values and
        for both passage answers and minimal answers.

        Returns:
          boolean flag whether it is null span or not.
        """

        if (self.start_byte_offset < 0 and self.end_byte_offset < 0):
            return True
        return False

    def __str__(self):
        return '({},{})'.format(self.start_byte_offset, self.end_byte_offset)

    def __repr__(self):
        return self.__str__()


def safe_divide(x, y):
    """Compute x / y, but return 0 if y is zero."""
    if y == 0:
        return 0
    else:
        return x / y


def safe_average(elements):
    """Computes average `elements`, but returns 0 if `elements` is empty."""
    return safe_divide(sum(elements), len(elements))


def compute_partial_match_scores(gold_span, pred_span):
    """Compute byte indices precision, recall and F1 score between span a and b.

    This is used for scoring only minimal answers. See `nonnull_span_equal` for
    scoring passage answers.

    Args:
      gold_span: a Span object. End_byte is inclusive (start_byte+byte_len)
      pred_span: a Span object.  Only compare non-null spans.
        Then, if the bytes are ot negative, compare byte offsets.

    Returns:
      precision: byte offset based precision.
                (# bytes in both gold and pred span) / (# bytes in pred_span)
      recall: byte offset based recall.
                (# bytes in both gold and pred span) / (# bytes in gold_span)
      f1: harmonic mean of precision and recall.
    """
    if not isinstance(gold_span, Span):
        raise TypeError('Gold span must has a Span type.')
    # if not isinstance(pred_span, NQSpan):
    #     raise TypeError('Prediction span must has a Span type.')
    if not isinstance(pred_span, Span):
        raise TypeError('Prediction span must has a Span type.')
    if gold_span.is_null_span():
        raise ValueError(
            'Null gold span should not be passed for F1 computation.')
    if pred_span.is_null_span():
        raise ValueError(
            'Null prediction span should not be passed for F1 computation.')
    assert not pred_span.is_null_span()
    # If there is no overlap, partial score is zero.
    if ((gold_span.end_byte_offset <= pred_span.start_byte_offset) or
            (pred_span.end_byte_offset <= gold_span.start_byte_offset)):
        precision = 0.0
        recall = 0.0

    else:
        in_both = (min(gold_span.end_byte_offset, pred_span.end_byte_offset) -
                   max(gold_span.start_byte_offset, pred_span.start_byte_offset))
        assert in_both > 0
        # if gold span starts earlier than pred span.
        if gold_span.start_byte_offset <= pred_span.start_byte_offset:
            only_in_gold = pred_span.start_byte_offset - gold_span.start_byte_offset
            only_in_gold += max(0,
                                gold_span.end_byte_offset - pred_span.end_byte_offset)
            only_in_pred = max(pred_span.end_byte_offset - gold_span.end_byte_offset,
                               0)
        # if pred span starts earlier than gold span.
        else:
            only_in_pred = gold_span.start_byte_offset - pred_span.start_byte_offset
            only_in_pred += max(0,
                                pred_span.end_byte_offset - gold_span.end_byte_offset)
            only_in_gold = max(gold_span.end_byte_offset - pred_span.end_byte_offset,
                               0)
        precision = safe_divide(in_both, (in_both + only_in_pred))
        recall = safe_divide(in_both, (in_both + only_in_gold))

    f1 = safe_divide(2 * precision * recall, precision + recall)
    return precision, recall, f1


def nonnull_span_equal(span_a, span_b):
    """Given two spans, return if they are equal.

    This is used for scoring only passage answers.
    See `compute_partial_match_scores` for minimal answers.

    Args:
      span_a: a Span object.
      span_b: a Span object.  Only compare non-null spans. First, if the bytes are
        not negative, compare byte offsets.

    Returns:
      True or False
    """
    assert isinstance(span_a, Span)
    assert isinstance(span_b, Span)
    assert not span_a.is_null_span()
    assert not span_b.is_null_span()

    # if byte offsets are not negative, compare byte offsets
    if ((span_a.start_byte_offset >= 0 and span_a.end_byte_offset >= 0) and
            (span_b.start_byte_offset >= 0 and span_b.end_byte_offset >= 0)):

        if ((span_a.start_byte_offset == span_b.start_byte_offset) and
                (span_a.end_byte_offset == span_b.end_byte_offset)):
            return True

    return False


def gold_has_minimal_answer(gold_label_list, minimal_non_null_threshold):
    """Gets vote from annotators for judging if there is a minimal answer."""
    #  We consider if there is a minimal answer if there is an minimal answer span
    #  or the yes/no answer is not none.
    gold_has_answer = gold_label_list and sum([
        ((not label.minimal_answer_span.is_null_span()) or
         (label.yes_no_answer != 'none')) for label in gold_label_list
    ]) >= minimal_non_null_threshold

    return bool(gold_has_answer)


def gold_has_passage_answer(gold_label_list, passage_non_null_threshold):
    """Gets vote from annotators for judging if there is a passage answer."""

    gold_has_answer = gold_label_list and (sum([
        label.passage_answer_index >= 0  # passage answer not null
        for label in gold_label_list  # for each annotator
    ]) >= passage_non_null_threshold)

    return bool(gold_has_answer)

# def read_prediction_jsonl(predictions_path):
#     """Read the prediction jsonl file with scores.
#
#     Args:
#       predictions_path: the path for the jsonl prediction file.
#
#     Returns:
#       A dictionary with key = example_id, value = TyDiLabel.
#
#     """
#     logging.info('Reading predictions from file: %s', format(predictions_path))
#     predictions = []
#     with open(predictions_path, 'r') as f:
#         for line in f:
#             predictions.append(json.loads(line))
#
#     tydi_pred_dict = {}
#     for single_prediction in predictions:
#
#         if 'passage_answer_index' in single_prediction:
#             passage_answer_index = single_prediction['passage_answer_index']
#         else:
#             passage_answer_index = -1
#
#         if 'minimal_answer' in single_prediction:
#             minimal_span = Span(
#                 single_prediction['minimal_answer']['start_byte_offset'],
#                 single_prediction['minimal_answer']['end_byte_offset'])
#
#         yes_no_answer = 'none'
#         if 'yes_no_answer' in single_prediction:
#             yes_no_answer = single_prediction['yes_no_answer'].lower()
#             if yes_no_answer not in ['yes', 'no', 'none']:
#                 raise ValueError('Invalid yes_no_answer value in prediction')
#
#             if yes_no_answer != 'none' and not minimal_span.is_null_span():
#                 raise ValueError(
#                     'yes/no prediction and minimal answers cannot coexist.')
#
#         pred_item = TyDiLabel(
#             example_id=single_prediction['example_id'],
#             language=single_prediction['language'],
#             passage_answer_index=passage_answer_index,
#             minimal_answer_span=minimal_span,
#             yes_no_answer=yes_no_answer,
#             question_text='',
#             plaintext='',
#             passage_score=float(single_prediction['passage_answer_score']),
#             minimal_score=float(single_prediction['minimal_answer_score']))
#
#         tydi_pred_dict[single_prediction['example_id']] = pred_item
#
#     return tydi_pred_dict


# def read_annotation_from_file(input_file):
#     """Read annotation from file."""
#     annotation_dict = {}
#     for line in input_file:
#         json_example = json.loads(line)
#         example_id = json_example['example_id']
#
#         # There are three annotations for each TyDi QA evaluation example.
#         annotation_list = []
#         raw_annotations = json_example.get('annotations', [])
#         if not raw_annotations:
#             logging.warning("No annotations found (may be test data) for example {} -- creating label without annotations".format(example_id))
#             #This is a work-around for running predictions when there are no annotations
#             #We only need the document_text.  TODO: find better way to handle this
#             gold_label = TyDiLabel(
#                 example_id=example_id,
#                 language=json_example['language'],
#                 passage_answer_index=None,
#                 passage_span=None,
#                 minimal_answer_span=None,
#                 passage_score=0,
#                 minimal_score=0,
#                 question_text=json_example['question_text'],
#                 plaintext=json_example['document_plaintext'],
#                 yes_no_answer=None)
#             annotation_list.append(gold_label)
#             annotation_dict[example_id] = annotation_list
#             continue
#         else:
#             logging.debug("Processing {} annotations for example {}".format(len(raw_annotations), example_id))
#
#         for annotation in raw_annotations:
#             passage_span_ind = annotation['passage_answer']['candidate_index']
#             if passage_span_ind == -1:
#                 passage_span = None
#             else:
#                 passage_span_par_offsets = json_example['passage_answer_candidates'][passage_span_ind]
#                 passage_span = Span(passage_span_par_offsets['plaintext_start_byte'],
#                                     passage_span_par_offsets['plaintext_end_byte'])
#             minimal_span_dict = annotation['minimal_answer']
#             minimal_span = Span(minimal_span_dict['plaintext_start_byte'],
#                                 minimal_span_dict['plaintext_end_byte'])
#
#             gold_label = TyDiLabel(
#                 example_id=example_id,
#                 language=json_example['language'],
#                 passage_answer_index=passage_span_ind,
#                 passage_span=passage_span,
#                 minimal_answer_span=minimal_span,
#                 passage_score=0,
#                 minimal_score=0,
#                 question_text=json_example['question_text'],
#                 plaintext=json_example['document_plaintext'],
#                 yes_no_answer=annotation['yes_no_answer'].lower())
#
#             annotation_list.append(gold_label)
#         annotation_dict[example_id] = annotation_list
#
#     return annotation_dict


# def read_annotation(path_name, n_threads=10):
#     """Read annotations with real multiple processes."""
#     input_paths = glob.glob(path_name)
#     pool = multiprocessing.Pool(n_threads)
#     logging.debug("Reading annotation from: {}".format(path_name))
#     try:
#         dict_list = pool.map(read_annotation_from_one_split, input_paths)
#     finally:
#         pool.close()
#         pool.join()
#
#     final_dict = {}
#     for single_dict in dict_list:
#         final_dict.update(single_dict)
#
#     logging.debug("Read annotation (from {}): {}".format(path_name, final_dict))
#
#     return final_dict


# def read_annotation_from_one_split(path_name):
#     """Read annotations from path_name."""
#     if '.gz' in path_name:
#         logging.info('Parsing %s (gzip)...', path_name)
#         with gzip.GzipFile(path_name, 'r') as input_file:
#             return read_annotation_from_file(input_file)
#
#     logging.info('Parsing %s...', path_name)
#     with open(path_name) as input_file:
#         return read_annotation_from_file(input_file)
