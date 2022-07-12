from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import glob
import json
import logging
import multiprocessing
from gzip import GzipFile

from functools import partial
from typing import Dict, Set, Optional, List


class InconsistentSpanError(ValueError):
    pass


@functools.total_ordering
class NQSpan(object):
    """A class for handling token and byte spans.
       Taken from https://github.com/google-research-datasets/natural-questions/blob/master/eval_utils.py#L60
      The logic is:
      1) if both start_byte !=  -1 and end_byte != -1 then the span is defined
         by byte offsets
      2) else, if start_token != -1 and end_token != -1 then the span is define
         by token offsets
      3) else, this is a null span.
      Null spans means that there is no (long or short) answers.
      If your systems only care about token spans rather than byte spans, set all
      byte spans to -1.
    """
    __slots__ = ['start_byte', 'end_byte', 'start_token', 'end_token', 'score', 'long_score']

    def __init__(self, start_byte: int, end_byte: int, start_token: int, end_token: int,
                 score: Optional[float] = None, long_score: Optional[float] = None, enforce_byte_consistency: bool = True,
                 enforce_token_consistency: bool = True):

        self.start_byte = start_byte
        self.end_byte = end_byte
        self.start_token = start_token
        self.end_token = end_token
        self.score = score
        self.long_score = long_score

        if enforce_byte_consistency:
            if ((start_byte < 0 <= end_byte) or
                    (start_byte >= 0 > end_byte)):
                raise InconsistentSpanError('Inconsistent Null Spans (Byte): %s' % self)

            if start_byte >= 0 and 0 <= end_byte < start_byte:
                raise InconsistentSpanError('Invalid byte spans (start_byte > end_byte): %s' % self)

        if enforce_token_consistency:
            if ((start_token < 0 <= end_token) or
                    (start_token >= 0 > end_token)):
                raise InconsistentSpanError('Inconsistent Null Spans (Token): %s' % self)

            if ((start_token >= 0 and end_token >= 0) and
                    (start_token > end_token)):
                raise InconsistentSpanError(
                    'Invalid token spans (start_token_idx > end_token_idx): %s' % self)

    def is_null_span(self):
        """A span is a null span if the start and end are both -1."""

        if (self.start_byte < 0 and self.end_byte < 0 and
                    self.start_token < 0 and self.end_token < 0):
            return True
        return False

    @staticmethod
    def null_span():
        return NQSpan(start_byte=-1, end_byte=-1, start_token=-1, end_token=-1)

    def contains(self, other_span):

        if other_span.start_byte > -1 and self.start_byte > -1 and other_span.end_byte > -1 and self.end_byte > -1:
            # Compare using bytes
            if other_span.start_byte >= self.start_byte and other_span.end_byte <= self.end_byte:
                return True
            else:
                return False
        elif other_span.start_token > -1 and self.start_token > -1 and other_span.end_token > -1 and self.end_token > -1:
            # Compare using tokens
            if other_span.start_token >= self.start_token and other_span.end_token <= self.end_token:
                return True
            else:
                return False
        else:
            return False

    def __str__(self):
        byte_str = 'bytes: [' + str(self.start_byte) + ',' + str(self.end_byte) + ')'
        tok_str = ('tokens: [' + str(self.start_token) + ',' + str(
            self.end_token) + ')')
        if self.score is not None:
            score_str = ', score: %s' % self.score
        else:
            score_str = ''

        return "Span(" + byte_str + ', ' + tok_str + score_str + ")"

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash((self.start_byte, self.end_byte, self.start_token, self.end_token, self.score))

    def __lt__(self, other):
        if self.score < other.score:
            return True
        return False

    def __eq__(self, o: object) -> bool:

        if not isinstance(o, NQSpan):
            return False

        for attribute in self.__slots__:
            if not hasattr(o, attribute):
                return False
            elif getattr(o, attribute) != getattr(self, attribute):
                return False

        return True


# A data structure for storing prediction and annotation.
# When a example has multiple annotations, multiple NQLabel will be used.
NQLabel = collections.namedtuple(
    'NQLabel',
    [
        'example_id',  # the unique id for each NQ example.
        'long_answer_span',  # A Span object for long answer.
        'short_answer_span_list',  # A list of Spans for short answer.
        #   Note that In NQ, the short answers
        #   do not need to be in a single span.
        'yes_no_answer',  # Indicate if the short answer is an yes/no answer
        #   The possible values are "yes", "no", "none".
        #   (case insensitive)
        #   If the field is "yes", short_answer_span_list
        #   should be empty or only contain null spans.
        'long_score',  # The prediction score for the long answer prediction.
        'short_score'  # The prediction score for the short answer prediction.
    ])


def is_null_span_list(span_list):
    """Returns true iff all spans in span_list are null or span_list is empty."""
    if not span_list or all([span.is_null_span() for span in span_list]):
        return True
    return False


def nonnull_span_equal(span_a: NQSpan, span_b: NQSpan):
    """Given two spans, return if they are equal.

    Args:
      span_a: a Span object.
      span_b: a Span object.  Only compare non-null spans. First, if the bytes are
        not negative, compare byte offsets, Otherwise, compare token offsets.

    Returns:
      True or False
    """
    assert isinstance(span_a, NQSpan)
    assert isinstance(span_b, NQSpan)
    assert not span_a.is_null_span()
    assert not span_b.is_null_span()

    # if byte offsets are not negative, compare byte offsets
    if ((span_a.start_byte >= 0 and span_a.end_byte >= 0) and
            (span_b.start_byte >= 0 and span_b.end_byte >= 0)):

        if ((span_a.start_byte == span_b.start_byte) and
                (span_a.end_byte == span_b.end_byte)):
            return True

    # if token offsets are not negative, compare token offsets
    if ((span_a.start_token >= 0 and span_a.end_token >= 0) and
            (span_b.start_token >= 0 and span_b.end_token >= 0)):

        if ((span_a.start_token == span_b.start_token) and
                (span_a.end_token == span_b.end_token)):
            return True

    return False


def span_set_equal(gold_span_list, pred_span_list):
    """Make the spans are completely equal besides null spans."""

    gold_span_list = [span for span in gold_span_list if not span.is_null_span()]
    pred_span_list = [span for span in pred_span_list if not span.is_null_span()]

    for pspan in pred_span_list:
        # not finding pspan equal to any spans in gold_span_list
        if not any([nonnull_span_equal(pspan, gspan) for gspan in gold_span_list]):
            return False

    for gspan in gold_span_list:
        # not finding gspan equal to any spans in pred_span_list
        if not any([nonnull_span_equal(pspan, gspan) for pspan in pred_span_list]):
            return False

    return True


def gold_has_short_answer(gold_label_list: list, short_non_null_threshold: int = 2) -> bool:
    """
    Gets vote from multi-annotators for judging if there is a short answer.
    :param gold_label_list: list of gold labels
    :param short_non_null_threshold: Require this many non-null short answer annotations
      to count gold as containing a short answer. Defaults to 2 like the original paper.
    """

    if not gold_label_list:
        return False  # Empty list will not have gold short answer

    # We consider if there is a short answer if there is an short answer span or
    #  the yes/no answer is not none.
    gold_has_answer = sum([
        ((not is_null_span_list(label.short_answer_span_list)) or
         (label.yes_no_answer != 'none')) for label in gold_label_list
    ]) >= short_non_null_threshold

    return gold_has_answer


def gold_has_long_answer(gold_label_list: list, long_non_null_threshold: int = 2) -> bool:
    """
    Gets vote from multi-annotators for judging if there is a long answer.
    :param gold_label_list: list of gold labels for judging
    :param long_non_null_threshold: Require this many non-null long answer annotations
      to count gold as containing a long answer. Defaults to 2 like the original paper.
    """

    if not gold_label_list:
        return False  # Empty list will not have gold long answer

    gold_has_answer = (sum([
        not label.long_answer_span.is_null_span()  # long answer not null
        for label in gold_label_list  # for each annotator
    ]) >= long_non_null_threshold)

    return gold_has_answer


def read_prediction_json_from_file(predictions_path, examples_to_filter_for=None):
    """Read the prediction json with scores.

    Args:
      predictions_path: the path for the prediction json.
      examples_to_filter_for: a set of examples to filter for (can be None, in which
       case all examples are kept)

    Returns:
      A dictionary with key = example_id, value = NQInstancePrediction.

    """
    logging.info('Reading predictions from file: %s', format(predictions_path))
    with open(predictions_path, 'r') as f:
        predictions = json.loads(f.read())

    return parse_json_as_predictions(predictions, examples_to_filter_for)


def parse_json_as_topk_predictions(
        predictions: dict, examples_to_filter_for: Optional[Set] = None) -> \
        Dict[int, List[NQLabel]]:

    nq_pred_dict = {}
    for topk_predictions_for_single_example in predictions["top_k_best_predictions"]:
        example_id = None
        topk_predictions = list()
        for single_prediction in topk_predictions_for_single_example:
            prediction = _parse_single_prediction_json(single_prediction)
            if example_id is None:
                example_id = prediction.example_id
                if examples_to_filter_for is not None and example_id not in examples_to_filter_for:
                    logging.debug('Skipping example %s' % example_id)
                    break
                else:
                    example_id = single_prediction['example_id']

            topk_predictions.append(prediction)

        if topk_predictions:
            nq_pred_dict[example_id] = topk_predictions

    return nq_pred_dict


def _parse_single_prediction_json(single_prediction: dict) -> NQLabel:
    if 'long_answer' in single_prediction:
        long_span = NQSpan(single_prediction['long_answer']['start_byte'],
                           single_prediction['long_answer']['end_byte'],
                           single_prediction['long_answer']['start_token'],
                           single_prediction['long_answer']['end_token'])
    else:
        long_span = NQSpan.null_span()  # Span is null if not presented.

    short_span_list = []
    if 'short_answers' in single_prediction:
        for short_item in single_prediction['short_answers']:
            short_span_list.append(
                NQSpan(short_item['start_byte'], short_item['end_byte'],
                       short_item['start_token'], short_item['end_token']))

    yes_no_answer = 'none'
    if 'yes_no_answer' in single_prediction:
        yes_no_answer = single_prediction['yes_no_answer'].lower()
        if yes_no_answer not in ['yes', 'no', 'none']:
            raise ValueError('Invalid yes_no_answer value in prediction')

        if yes_no_answer != 'none' and not is_null_span_list(short_span_list):
            raise ValueError('yes/no prediction and short answers cannot coexist.')

    return NQLabel(
        example_id=single_prediction['example_id'],
        long_answer_span=long_span,
        short_answer_span_list=short_span_list,
        yes_no_answer=yes_no_answer,
        long_score=float(single_prediction['long_answer_score']),
        short_score=float(single_prediction['short_answers_score']))


def parse_json_as_predictions(predictions: dict, examples_to_filter_for: Optional[Set] = None) -> \
        Dict[int, NQLabel]:
    nq_pred_dict = {}

    for single_prediction in predictions['predictions']:
        if examples_to_filter_for is None or \
                        single_prediction['example_id'] in examples_to_filter_for:
            nq_pred_dict[single_prediction['example_id']] = _parse_single_prediction_json(
                single_prediction)

    return nq_pred_dict


def read_annotation_from_one_split(gzipped_input_file, example_ids_to_filter_by=None):
    """Read annotation from one split of file."""
    if example_ids_to_filter_by and not isinstance(example_ids_to_filter_by, set):
        example_ids_to_filter_by = set(example_ids_to_filter_by)

    if isinstance(gzipped_input_file, str):
        gzipped_input_file = open(gzipped_input_file, mode='rb')
    logging.info('parsing %s ..... ', gzipped_input_file.name)
    annotation_dict = {}
    with GzipFile(fileobj=gzipped_input_file) as input_file:
        for line in input_file:
            json_example = json.loads(line)
            example_id = json_example['example_id']

            if not example_ids_to_filter_by or example_id in example_ids_to_filter_by:

                # There are multiple annotations for one nq example.
                annotation_list = []

                for annotation in json_example['annotations']:
                    long_span_rec = annotation['long_answer']
                    long_span = NQSpan(long_span_rec['start_byte'], long_span_rec['end_byte'],
                                       long_span_rec['start_token'],
                                       long_span_rec['end_token'])

                    short_span_list = []
                    for short_span_rec in annotation['short_answers']:
                        short_span = NQSpan(
                            short_span_rec['start_byte'], short_span_rec['end_byte'],
                            short_span_rec['start_token'], short_span_rec['end_token'])
                        short_span_list.append(short_span)

                    gold_label = NQLabel(
                        example_id=example_id,
                        long_answer_span=long_span,
                        short_answer_span_list=short_span_list,
                        long_score=0,
                        short_score=0,
                        yes_no_answer=annotation['yes_no_answer'].lower())

                    annotation_list.append(gold_label)
                annotation_dict[example_id] = annotation_list

    return annotation_dict


def read_annotation(path_name, n_threads=10, example_ids_to_filter_by=None,
                    read_from_split_fn=read_annotation_from_one_split):
    """Read annotations with real multiple processes."""
    input_paths = glob.glob(path_name)
    pool = multiprocessing.Pool(n_threads)
    logging.debug("Reading annotation from: {}".format(path_name))

    if example_ids_to_filter_by is not None:
        example_ids_to_filter_by = list(example_ids_to_filter_by)

    annotation_reader = partial(read_from_split_fn,
                                example_ids_to_filter_by=example_ids_to_filter_by)
    try:
        dict_list = pool.map(annotation_reader, input_paths)
    finally:
        pool.close()
        pool.join()

    final_dict = {}
    for single_dict in dict_list:
        final_dict.update(single_dict)

    logging.debug("Read annotation (from {}): {}".format(path_name, final_dict))

    return final_dict
