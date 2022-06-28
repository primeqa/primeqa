from typing import Dict, Any, Tuple, List

import datasets

from primeqa.mrc.metrics.tydi_f1.eval_utils import Span, TyDiLabel
from primeqa.mrc.metrics.tydi_f1.tydi_eval import pretty_print
from primeqa.mrc.data_models.target_type import TargetType


_DESCRIPTION = """
The F1 score is the harmonic mean of the precision and recall. It can be computed with:
F1 = 2 * (precision * recall) / (precision + recall).  This implementation of F1 is based
on the TyDi QA leaderboard.

Adapted from https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels.
    references: Ground truth labels.
    passage_non_null_threshold: threshold for number of null annotations annotations to consider the passage answer as null (default=2)
    span_non_null_threshold: threshold for number of null annotations annotations to consider the span answer as null (default=2)
    verbose: dump reference and prediction for debugging purposes
    
Returns: metrics dict comprising:

  * minimal_f1: Minimal Answer F1.
  * minimal_precision: Minimal Answer Precision.
  * minimal_recall: Minimal Answer Recall.
  * passage_f1: Passage Answer F1.
  * passage_precision: Passage Answer Precision.
  * passage_recall: Passage Answer Recall.
"""

_CITATION = """\
@article{tydiqa,
title   = {TyDi QA: A Benchmark for Information-Seeking Question Answering in Typologically Diverse Languages},
author  = {Jonathan H. Clark and Eunsol Choi and Michael Collins and Dan Garrette and Tom Kwiatkowski and 
           Vitaly Nikolaev and Jennimaria Palomaki}
year    = {2020},
journal = {Transactions of the Association for Computational Linguistics}
}
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
        document_plaintext=datasets.Value("string"),
        question=datasets.Value("string")
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
            reference_urls=["https://github.com/google-research-datasets/tydiqa/blob/master/tydi_eval.py"],
        )

    def _compute(self, *, predictions=None, references=None, passage_non_null_threshold=2, span_non_null_threshold=2, verbose=False, **kwargs) -> Dict[str, Any]:
        
        if not predictions:
            raise ValueError("No predictions provided")
        elif not references:
            raise ValueError("No references provided")

        predictions = dict(map(self._convert_pred_to_entry, predictions))
        references = dict(map(self._convert_ref_to_entry, references))

        metrics = pretty_print(references, predictions, passage_non_null_threshold=passage_non_null_threshold, span_non_null_threshold=span_non_null_threshold, verbose=verbose)
        return metrics

    def _convert_ref_to_entry(self, ref: dict) -> Tuple[str, List[TyDiLabel]]:
        """
        Converts a reference dict into an example_id, [labels] pair.
        """
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
                question_text=ref['question'][i],
                plaintext=ref['document_plaintext'][i],
            ) for i in range(len(ref['passage_index']))
        ]
        return key, value

    def _convert_pred_to_entry(self, pred: dict) -> Tuple[str, TyDiLabel]:
        """
        Converts a prediction dict into an example_id, label pair.
        """
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

    @staticmethod
    def _bool_target(target_type: TargetType) -> str:
        """
        Converts a target type into a boolean string as expected by TyDi eval.
        """
        if target_type == TargetType.YES:
            return 'yes'
        elif target_type == TargetType.NO:
            return 'no'
        elif target_type == TargetType.NO_ANSWER:
            return 'none'
        else:
            raise NotImplementedError(f"Unexpected target type for tydi bool string conversion: {target_type}")


