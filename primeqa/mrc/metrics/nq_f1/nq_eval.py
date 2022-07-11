from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import pickle
from collections import OrderedDict

# from iota.mrc.mnlp.io.cmd_arg_parser import positive_integer_type
from os import path
from typing import List, Union, Dict, Tuple, Optional

from primeqa.mrc.metrics.nq_f1 import eval_utils as util


def safe_divide(x, y):
    """Compute x / y, but return 0 if y is zero."""
    if y == 0:
        return 0
    else:
        return x / y


def score_long_answer(gold_label_list, pred_label, long_non_null_threshold=None):
    """Scores a long answer as correct or not.

    1) First decide if there is a gold long answer with LONG_NO_NULL_THRESHOLD.
    2) The prediction will get a match if:
       a. There is a gold long answer.
       b. The prediction span match exactly with *one* of the non-null gold
          long answer span.

    Args:
      gold_label_list: A list of NQLabel, could be None.
      pred_label: A single NQLabel, could be None.
      long_non_null_threshold: Min number of non null spans in the annotations to consider the
       question as having a non null answer

    Returns:
      gold_has_answer, pred_has_answer, is_correct, score
    """
    gold_has_answer_kwargs = dict(gold_label_list=gold_label_list)
    if long_non_null_threshold is not None:  # if not set omit kwarg to use default in util.gold_has_long_answer
        gold_has_answer_kwargs['long_non_null_threshold'] = long_non_null_threshold
    gold_has_answer = util.gold_has_long_answer(**gold_has_answer_kwargs)

    pred_has_answer = pred_label and (
        not pred_label.long_answer_span.is_null_span())

    is_correct = False
    score = pred_label.long_score

    # Both sides are non-null spans.
    if gold_has_answer and pred_has_answer:
        for gold_label in gold_label_list:
            # while the voting results indicate there is an long answer, each
            # annotator might still say there is no long answer.
            if gold_label.long_answer_span.is_null_span():
                continue

            if util.nonnull_span_equal(gold_label.long_answer_span,
                                       pred_label.long_answer_span):
                is_correct = True
                break

    return gold_has_answer, pred_has_answer, is_correct, score


def score_short_answer(gold_label_list, pred_label, short_non_null_threshold=None):
    """Scores a short answer as correct or not.

    1) First decide if there is a gold short answer with SHORT_NO_NULL_THRESHOLD.
    2) The prediction will get a match if:
       a. There is a gold short answer.
       b. The prediction span *set* match exactly with *one* of the non-null gold
          short answer span *set*.

    Args:
      gold_label_list: A list of NQLabel.
      pred_label: A single NQLabel.
      short_non_null_threshold: Min number of non null annotations required before considering
        the question as having a non null answer.  Optional.

    Returns:
      gold_has_answer, pred_has_answer, is_correct, score
    """

    # There is a gold short answer if gold_label_list not empty and non null
    # answers is over the threshold (sum over annotators).
    gold_has_answer_kwargs = dict(gold_label_list=gold_label_list)
    if short_non_null_threshold is not None:  # if not set omit kwarg to use default in util.gold_has_short_answer
        gold_has_answer_kwargs['short_non_null_threshold'] = short_non_null_threshold
    gold_has_answer = util.gold_has_short_answer(**gold_has_answer_kwargs)

    # There is a pred long answer if pred_label is not empty and short answer
    # set is not empty.
    pred_has_answer = pred_label and (
        (not util.is_null_span_list(pred_label.short_answer_span_list)) or
        pred_label.yes_no_answer != 'none')

    is_correct = False
    score = pred_label.short_score

    # Both sides have short answers, which contains yes/no questions.
    if gold_has_answer and pred_has_answer:
        if pred_label.yes_no_answer != 'none':  # System thinks its y/n questions.
            for gold_label in gold_label_list:
                if pred_label.yes_no_answer == gold_label.yes_no_answer:
                    is_correct = True
                    break
        else:
            for gold_label in gold_label_list:
                if util.span_set_equal(gold_label.short_answer_span_list,
                                       pred_label.short_answer_span_list):
                    is_correct = True
                    break

    return gold_has_answer, pred_has_answer, is_correct, score


def score_answers(gold_annotation_dict, pred_dict, skip_missing_example_ids: bool = False,
                  long_non_null_threshold: int = 2, short_non_null_threshold: int = 2):
    """Scores all answers for all documents.

    Args:
      gold_annotation_dict: a dict from example id to list of NQLabels.
      pred_dict: a dict from example id to list of NQLabels.
      skip_missing_example_ids: True to only use example ids from intersection of gold and preds
      long_non_null_threshold: Min number of non null spans in the annotation before considering
        the question to be requiring a non null answer
      short_non_null_threshold: Min number of non null spans in the annotation before considering
        the question to be one with a non null answer

    Returns:
      long_answer_stats: List of scores for long answers.
      short_answer_stats: List of scores for short answers.
    """
    gold_id_set = set(gold_annotation_dict.keys())
    pred_id_set = set(pred_dict.keys())
    sym_diff = gold_id_set.symmetric_difference(pred_id_set)

    if (not skip_missing_example_ids) and sym_diff:
        raise ValueError('ERROR: the example ids in gold annotations and example '
                         'ids in the prediction are not equal.')
    elif skip_missing_example_ids and sym_diff:
        logging.warning("Skipping {} example ids that are only in either gold or preds".format(len(sym_diff)))

    long_answer_stats = []
    short_answer_stats = []
    id_set = gold_id_set if not skip_missing_example_ids else gold_id_set.intersection(pred_id_set)

    for example_id in id_set:
        gold = gold_annotation_dict[example_id]
        pred = pred_dict[example_id]

        long_answer_stats.append(score_long_answer(gold_label_list=gold, pred_label=pred,
                                                   long_non_null_threshold=long_non_null_threshold))
        short_answer_stats.append(score_short_answer(gold_label_list=gold, pred_label=pred,
                                                     short_non_null_threshold=short_non_null_threshold))

    # use the 'score' column, which is last
    long_answer_stats.sort(key=lambda x: x[-1], reverse=True)
    short_answer_stats.sort(key=lambda x: x[-1], reverse=True)

    return long_answer_stats, short_answer_stats


def compute_f1(answer_stats, prefix=''):
    """Computes F1, precision, recall for a list of answer scores.

    Args:
      answer_stats: List of per-example scores.
      prefix (''): Prefix to prepend to score dictionary.

    Returns:
      Dictionary mapping string names to scores.
    """

    has_gold, has_pred, is_correct, _ = zip(*answer_stats)
    precision = safe_divide(sum(is_correct), sum(has_pred))
    recall = safe_divide(sum(is_correct), sum(has_gold))
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return OrderedDict({
        prefix + 'n': len(answer_stats),
        prefix + 'f1': f1,
        prefix + 'precision': precision,
        prefix + 'recall': recall
    })


def extract_metrics_at_optimal_threshold(answer_stats: list) -> Tuple[float, float, float, float]:
    """
    :param answer_stats: one of the dictionaries returned from score_answers
    :return: Tuple (f1, precision, recall, optimal_threshold)
    """
    opt_result, pr_table = compute_pr_curves(
        answer_stats, targets=[0.5, 0.75, 0.9])
    return opt_result # f1, precision, recall, optimal_threshold


def compute_optimal_metrics(long_answer_stats: List[List[Union[bool, float]]],
                            short_answer_stats: List[List[Union[bool, float]]]) -> Dict[str, float]:
    """Computes overall metrics for long and short answers for their respective optimal thresholds
    Arguments:
       long_answer_stats: List of long answer scores.
       short_answer_stats: List of short answer scores.
    Returns:
       Ordered Dictionary of name (string) -> score.
    """
    f1, precision, recall, optimal_threshold = extract_metrics_at_optimal_threshold(long_answer_stats)
    prefix = 'long-answer-'
    scores = OrderedDict({
        prefix + 'n': len(long_answer_stats),
        prefix + 'f1': f1,
        prefix + 'precision': precision,
        prefix + 'recall': recall,
        prefix + 'optimal-threshold': optimal_threshold
    })

    f1, precision, recall, optimal_threshold = extract_metrics_at_optimal_threshold(short_answer_stats)
    prefix = 'short-answer-'
    scores.update(OrderedDict({
        prefix + 'n': len(short_answer_stats),
        prefix + 'f1': f1,
        prefix + 'precision': precision,
        prefix + 'recall': recall,
        prefix + 'optimal-threshold': optimal_threshold
    }))
    return scores


def compute_final_f1(long_answer_stats, short_answer_stats):
    """Computes overall F1 given long and short answers, ignoring scores.

    Note: this assumes that the answers have been thresholded.

    Arguments:
       long_answer_stats: List of long answer scores.
       short_answer_stats: List of short answer scores.

    Returns:
       Dictionary of name (string) -> score.
    """
    scores = compute_f1(long_answer_stats, prefix='long-answer-')
    scores.update(compute_f1(short_answer_stats, prefix='short-answer-'))
    return scores


def compute_pr_curves(answer_stats, targets: Optional[List]=None):
    """Computes PR curve and returns R@P for specific targets.

    The values are computed as follows: find the (precision, recall) point
    with maximum recall and where precision > target.

    Arguments:
      answer_stats: List of statistic tuples from the answer scores.
      targets (None): List of precision thresholds to target.

    Returns:
      List of table with rows: [target, r, p, score].
    """
    try:
        total_correct = 0
        total_has_pred = 0
        total_has_gold = 0

        # Count the number of gold annotations.
        for has_gold, _, _, _ in answer_stats:
            total_has_gold += has_gold

        # Keep track of the point of maximum recall for each target.
        max_recall = [0 for _ in targets]
        max_precision = [0 for _ in targets]
        max_scores = [None for _ in targets]

        # Only keep track of unique thresholds in this dictionary.
        scores_to_stats = OrderedDict()

        # Loop through every possible threshold and compute precision + recall.
        for has_gold, has_pred, is_correct, score in answer_stats:
            total_correct += is_correct
            total_has_pred += has_pred

            precision = safe_divide(total_correct, total_has_pred)
            recall = safe_divide(total_correct, total_has_gold)

            # If there are any ties, this will be updated multiple times until the
            # ties are all counted.
            scores_to_stats[score] = [precision, recall]

        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_threshold = 0.0

        for threshold, (precision, recall) in scores_to_stats.items():
            # Match the thresholds to the find the closest precision above some target.
            for t, target in enumerate(targets):
                if precision >= target and recall > max_recall[t]:
                    max_recall[t] = recall
                    max_precision[t] = precision
                    max_scores[t] = threshold

            # Compute optimal threshold.
            f1 = safe_divide(2 * precision * recall, precision + recall)
            if f1 > best_f1:
                best_f1 = f1
                best_precision = precision
                best_recall = recall
                best_threshold = threshold

        return ((best_f1, best_precision, best_recall, best_threshold),
                zip(targets, max_recall, max_precision, max_scores))
    except Exception as ex:
        logging.error("Caught exception {} while computing p/r curve"
                      " for answers: {} and targets: {}".format(ex, answer_stats, targets))
        raise


def print_r_at_p_table(answer_stats):
    """Pretty prints the R@P table for default targets."""
    opt_result, pr_table = compute_pr_curves(
        answer_stats, targets=[0.5, 0.75, 0.9])
    f1, precision, recall, optimal_threshold = opt_result
    print('Optimal threshold: {:.5}'.format(optimal_threshold))
    print(' F1     /  P      /  R')
    print('{: >7.2%} / {: >7.2%} / {: >7.2%}'.format(f1, precision, recall))
    for target, recall, precision, threshold in pr_table:
        if threshold is not None:
            print('R@P={}: {:.2%} (actual p={:.2%}, score threshold={:.4})'.format(
                target, recall, precision, threshold))
        else:
            print('R@P={}: No possible threshold values satisfy this R@P'.format(target))


def get_metrics_as_dict(gold_path, prediction_path, num_threads=10):
    """Library version of the end-to-end evaluation.

    Arguments:
      gold_path: Path to the gzip JSON data. For multiple files, should be a glob
        pattern (e.g. "/path/to/files-*")
      prediction_path: Path to the JSON prediction data.
      num_threads (10): Number of threads to use when parsing multiple files.

    Returns:
      metrics: A dictionary mapping string names to metric scores.
    """

    nq_gold_dict = util.read_annotation(gold_path, n_threads=num_threads)
    nq_pred_dict = util.read_prediction_json_from_file(prediction_path)
    logging.debug("Loaded gold {}: {}".format(gold_path, nq_gold_dict))
    logging.debug("Loaded pred {}: {}".format(prediction_path, nq_pred_dict))

    long_answer_stats, short_answer_stats = score_answers(nq_gold_dict, nq_pred_dict)

    return get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)


def get_metrics_with_answer_stats(long_answer_stats, short_answer_stats):
    """Generate metrics dict using long and short answer stats."""

    def _get_metric_dict(answer_stats, prefix=''):
        """Compute all metrics for a set of answer statistics."""
        opt_result, pr_table = compute_pr_curves(
            answer_stats, targets=[0.5, 0.75, 0.9])
        f1, precision, recall, threshold = opt_result
        metrics = OrderedDict({
            'best-threshold-f1': f1,
            'best-threshold-precision': precision,
            'best-threshold-recall': recall,
            'best-threshold': threshold,
        })
        for target, recall, precision, _ in pr_table:
            metrics['recall-at-precision>={:.2}'.format(target)] = recall
            metrics['precision-at-precision>={:.2}'.format(target)] = precision

        # Add prefix before returning.
        return dict([(prefix + k, v) for k, v in metrics.items()])

    logging.debug("Computing p/r for long answer stats {}".format(long_answer_stats))
    metrics = _get_metric_dict(long_answer_stats, 'long-')
    logging.debug("Computing p/r for short answer stats {}".format(short_answer_stats))
    metrics.update(_get_metric_dict(short_answer_stats, 'short-'))
    return metrics


def load_gt_lookup_as_dict(ground_truth_gzip_file_pattern, num_workers,
                           read_from_split_fn=util.read_annotation_from_one_split):
    previously_cached_gt_lookup = '{0}{1}_gt_lookup_cache.pickle'.format(
        path.splitext(ground_truth_gzip_file_pattern)[0],
        ('_' + read_from_split_fn.__name__) if read_from_split_fn != util.read_annotation_from_one_split else '')
    if path.isfile(previously_cached_gt_lookup):
        logging.info('Loading ground truth lookup from previously cached: %s' %
                     previously_cached_gt_lookup)
        with open(previously_cached_gt_lookup, 'rb') as infile:
            nq_gold_dict = pickle.load(infile)
    else:
        logging.info('No previously cached lookup; so generating lookup from files matching'
                     ' pattern: %s' % ground_truth_gzip_file_pattern)
        nq_gold_dict = util.read_annotation(ground_truth_gzip_file_pattern, n_threads=num_workers,
                                            read_from_split_fn=read_from_split_fn)
        cache_to_pickle_file(nq_gold_dict, previously_cached_gt_lookup)

    logging.info('Read in gt lookup for %d examples into memory' % len(nq_gold_dict))
    return nq_gold_dict


def pretty_print(long_answer_stats, short_answer_stats):
    print('*' * 20)
    print('LONG ANSWER R@P TABLE:')
    logging.debug("Printing r@p table for long answer stats: {}".format(long_answer_stats))
    print_r_at_p_table(long_answer_stats)
    print('*' * 20)
    print('SHORT ANSWER R@P TABLE:')
    print_r_at_p_table(short_answer_stats)

    scores = compute_final_f1(long_answer_stats, short_answer_stats)
    print('*' * 20)
    print('METRICS IGNORING SCORES (n={}):'.format(scores['long-answer-n']))
    print('              F1     /  P      /  R')
    print('Long answer  {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['long-answer-f1'], scores['long-answer-precision'],
        scores['long-answer-recall']))
    print('Short answer {: >7.2%} / {: >7.2%} / {: >7.2%}'.format(
        scores['short-answer-f1'], scores['short-answer-precision'],
        scores['short-answer-recall']))

    scores = {name.replace('-', '_'): value for name, value in scores.items()}
    return scores
