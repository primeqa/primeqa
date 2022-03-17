from typing import Dict, Any

import datasets

from oneqa.mrc.metrics.tydi_f1.eval_utils import Span, TyDiLabel
from oneqa.mrc.metrics.tydi_f1.tydi_eval import pretty_print
from oneqa.mrc.types.target_type import TargetType

# TODO tydi f1 docs
_DESCRIPTION = """
TODO
The F1 score is the harmonic mean of the precision and recall. It can be computed with:
F1 = 2 * (precision * recall) / (precision + recall).  This implementation of F1 is based
on the Natural Questions (NQ) leaderboard and does not award partial credit unlike
the SQuAD-style F1.

Adapted from https://ai.google.com/research/NaturalQuestions's `nq_eval` script.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels.
    references: Ground truth labels.
Returns: TODO
    minmal_f1: minimal answer F1 score.
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
class TyDiF1(datasets.Metric):
    _common_answer_schema = dict(
        start_position=datasets.Value("int32"),
        end_position=datasets.Value("int32"),
        passage_index=datasets.Value("int32"),
        yes_no_answer=datasets.Value("int32"),
        example_id=datasets.Value("string"),
    )
    _pred_answer_schema = dict(
        confidence_score=datasets.Value("float32"),
    )
    _ref_answer_schema = dict(
        language=datasets.Value("string"),
    )

    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                dict(
                    predictions={**self._common_answer_schema, **self._pred_answer_schema},
                    references=datasets.Sequence(feature={**self._common_answer_schema, **self._ref_answer_schema})
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

        predictions = dict(map(self._convert_pred_to_entry, predictions))
        references = dict(map(self._convert_ref_to_entry, references))

        metrics = pretty_print(references, predictions)
        return metrics

    def _convert_ref_to_entry(self, ref):
        if not all(ref['example_id'][0] == ref['example_id'][i] for i in range(len(ref['example_id']))):
            raise ValueError("Found mismatched examples")
        elif not all(ref['language'][0] == ref['language'][i] for i in range(len(ref['language']))):
            raise ValueError("Found mismatched languages")

        key = ref['example_id'][0]
        value = [
            TyDiLabel(
                example_id=ref['example_id'][i],
                passage_answer_index=ref['passage_index'][i],
                minimal_answer_span=Span(
                    ref['start_position'][i],
                    ref['end_position'][i])
                ,
                yes_no_answer=self._bool_target(
                    TargetType(ref['yes_no_answer'][i])
                ),
                passage_score=0,
                minimal_score=0,
                language=ref['language'][i],
                passage_span=None,
                question_text='',
                plaintext='',
            ) for i in range(len(ref['passage_index']))
        ]
        return key, value

    def _convert_pred_to_entry(self, pred):
        key = pred['example_id']
        value = TyDiLabel(
                example_id=pred['example_id'],
                passage_answer_index=pred['passage_index'],
                minimal_answer_span=Span(
                    pred['start_position'],
                    pred['end_position'])
                ,
                yes_no_answer=self._bool_target(
                    TargetType(pred['yes_no_answer'])
                ),
                passage_score=pred['confidence_score'] ,
                minimal_score=pred['confidence_score'] ,
                language=None,
                passage_span=None,
                question_text='',
                plaintext='',
            )
        return key, value

    def _bool_target(self, target_type: TargetType) -> str:
        if target_type == TargetType.YES:
            return 'yes'
        elif target_type == TargetType.NO:
            return 'no'
        elif target_type == TargetType.NO_ANSWER:
            return 'none'
        else:
            raise NotImplementedError(f"Unexpected target type for tydi bool string conversion: {target_type}")


