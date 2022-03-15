import logging
from collections import OrderedDict
from operator import itemgetter
from typing import Dict, Any

import datasets

from oneqa.mrc.metrics.nq_f1 import eval_utils as util
from oneqa.mrc.metrics.nq_f1.eval_utils import NQLabel

_DESCRIPTION = """
The F1 score is the harmonic mean of the precision and recall. It can be computed with:
F1 = 2 * (precision * recall) / (precision + recall).  This implementation of F1 is based
on the Natural Questions (NQ) leaderboard and does not award partial credit unlike
the SQuAD-style F1.

Adapted from https://ai.google.com/research/NaturalQuestions's `nq_eval` script.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels, as returned by a model.
    references: Ground truth labels.
Returns:
    span_f1: span answer F1 score.
    passage_f1: passage F1 score.
Examples:  TODO
    >>> f1_metric = datasets.load_metric("f1")
    >>> results = f1_metric.compute(predictions=[0, 1], references=[0, 1])
    >>> print(results)
    {'f1': 1.0}
    >>> predictions = [0, 2, 1, 0, 0, 1]
    >>> references = [0, 1, 2, 0, 1, 2]
    >>> results = f1_metric.compute(predictions=predictions, references=references, average="macro")
    >>> print(results)
    {'f1': 0.26666666666666666}
    >>> results = f1_metric.compute(predictions=predictions, references=references, average="micro")
    >>> print(results)
    {'f1': 0.3333333333333333}
    >>> results = f1_metric.compute(predictions=predictions, references=references, average="weighted")
    >>> print(results)
    {'f1': 0.26666666666666666}
    >>> results = f1_metric.compute(predictions=predictions, references=references, average=None)
    >>> print(results)
    {'f1': array([0.8, 0. , 0. ])}
"""

_CITATION = """\
TODO
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NQF1(datasets.Metric):
    _answer_schema = dict(
        start_position=datasets.Value("int32"),
        end_position=datasets.Value("int32"),
        passage_index=datasets.Value("int32"),
        yes_no_answer=datasets.Value("int32"),
        confidence_score=datasets.Value("float32"),
        example_id=datasets.Value("str"),
    )

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                dict(
                    predictions=self._answer_schema,
                    references=datasets.Sequence(feature=self._answer_schema)
                )),
            reference_urls=["https://github.com/google-research-datasets/natural-questions/blob/master/nq_eval.py"],
        )

    def _compute(self, *, predictions=None, references=None, **kwargs) -> Dict[str, Any]:
        if kwargs:
            raise ValueError(f"Unexpected kwargs: {kwargs}")
        elif not predictions:
            raise ValueError("No predictions provided")
        elif not references:
            raise ValueError("No references provided")

        predictions = self._convert_to_nq_labels(predictions)
        references = self._convert_to_nq_labels(references)

        long_answer_stats, short_answer_stats = self._score_answers(
            gold_references=references, predictions=predictions)
        metrics = self._get_metrics_with_answer_stats(long_answer_stats, short_answer_stats)
        return metrics

    def _convert_to_nq_labels(self, values):
        values = [
            NQLabel(example_id=value['example_id'],
                    long_answer_span=value['passage_index'],


                    ) for value in values
        ]
        return values

    # @staticmethod
    # def score_answers(gold_annotation_dict, pred_dict, skip_missing_example_ids: bool = False,
    #                   long_non_null_threshold: int = 2, short_non_null_threshold: int = 2):
    #     """Scores all answers for all documents.
    #
    #     Args:
    #       gold_annotation_dict: a dict from example id to list of NQLabels.
    #       pred_dict: a dict from example id to list of NQLabels.
    #       skip_missing_example_ids: True to only use example ids from intersection of gold and preds
    #       long_non_null_threshold: Min number of non null spans in the annotation before considering
    #         the question to be requiring a non null answer
    #       short_non_null_threshold: Min number of non null spans in the annotation before considering
    #         the question to be one with a non null answer
    #
    #     Returns:
    #       long_answer_stats: List of scores for long answers.
    #       short_answer_stats: List of scores for short answers.
    #     """
    #     gold_id_set = set(gold_annotation_dict.keys())
    #     pred_id_set = set(pred_dict.keys())
    #     sym_diff = gold_id_set.symmetric_difference(pred_id_set)
    #
    #     if (not skip_missing_example_ids) and sym_diff:
    #         raise ValueError('ERROR: the example ids in gold annotations and example '
    #                          'ids in the prediction are not equal.')
    #     elif skip_missing_example_ids and sym_diff:
    #         logging.warning("Skipping {} example ids that are only in either gold or preds".format(len(sym_diff)))
    #
    #     long_answer_stats = []
    #     short_answer_stats = []
    #     id_set = gold_id_set if not skip_missing_example_ids else gold_id_set.intersection(pred_id_set)
    #
    #     for example_id in id_set:
    #         gold = gold_annotation_dict[example_id]
    #         pred = pred_dict[example_id]
    #
    #         long_answer_stats.append(score_long_answer(gold_label_list=gold, pred_label=pred,
    #                                                    long_non_null_threshold=long_non_null_threshold))
    #         short_answer_stats.append(score_short_answer(gold_label_list=gold, pred_label=pred,
    #                                                      short_non_null_threshold=short_non_null_threshold))
    #
    #     # use the 'score' column, which is last
    #     long_answer_stats.sort(key=itemgetter(-1), reverse=True)
    #     short_answer_stats.sort(key=itemgetter(-1), reverse=True)
    #
    #     return long_answer_stats, short_answer_stats

    @classmethod
    def _score_answers(cls, gold_references, predictions, skip_missing_example_ids: bool = False,
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
        if len(gold_references) != len(predictions):
            raise ValueError(f"Expected equal lengths of gold ({len(gold_references)}) "
                             f"and predictions ({len(predictions)}).")

        long_answer_stats = []
        short_answer_stats = []

        for gold, pred in zip(gold_references, predictions):
            long_answer_stats.append(cls._score_long_answer(gold_label_list=gold, pred_label=pred,
                                                            long_non_null_threshold=long_non_null_threshold))
            short_answer_stats.append(cls._score_short_answer(gold_label_list=gold, pred_label=pred,
                                                              short_non_null_threshold=short_non_null_threshold))

        # use the 'score' column, which is last
        long_answer_stats.sort(key=itemgetter(-1), reverse=True)
        short_answer_stats.sort(key=itemgetter(-1), reverse=True)

        return long_answer_stats, short_answer_stats

    @classmethod
    def _score_long_answer(cls, gold_label_list, pred_label, long_non_null_threshold=None):
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

    @classmethod
    def _score_short_answer(cls, gold_label_list, pred_label, short_non_null_threshold=None):
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

    @classmethod
    def _get_metrics_with_answer_stats(cls, long_answer_stats, short_answer_stats):
        """Generate metrics dict using long and short answer stats."""

        def _get_metric_dict(answer_stats, prefix=''):
            """Compute all metrics for a set of answer statistics."""
            opt_result, pr_table = cls._compute_pr_curves(
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
        metrics = _get_metric_dict(long_answer_stats, 'passage-')
        logging.debug("Computing p/r for short answer stats {}".format(short_answer_stats))
        metrics.update(_get_metric_dict(short_answer_stats, 'span-'))
        return metrics

    @classmethod
    def _compute_pr_curves(cls, answer_stats, targets: Optional[List] = None):
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

                precision = cls._safe_divide(total_correct, total_has_pred)
                recall = cls._safe_divide(total_correct, total_has_gold)

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
                f1 = cls._safe_divide(2 * precision * recall, precision + recall)
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

    @staticmethod
    def _safe_divide(x, y):
        """Compute x / y, but return 0 if y is zero."""
        if y == 0:
            return 0
        else:
            return x / y
