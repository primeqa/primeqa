from typing import Dict, Any, Tuple, List

import datasets

from primeqa.mrc.metrics.nq_f1.eval_utils import NQLabel, NQSpan
from primeqa.mrc.metrics.nq_f1.nq_eval import pretty_print, get_metrics_with_answer_stats, score_answers
from primeqa.mrc.data_models.target_type import TargetType


_DESCRIPTION = """
The F1 score is the harmonic mean of the precision and recall. It can be computed with:
F1 = 2 * (precision * recall) / (precision + recall).
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Predicted labels.
    references: Ground truth labels.

Returns: metrics dict comprising:

  * LONG ANSWER R@P TABLE.
  * SHORT ANSWER R@P TABLE.
"""

_CITATION = """\
@article{47761,
title	= {Natural Questions: a Benchmark for Question Answering Research},
author	= {Tom Kwiatkowski and Jennimaria Palomaki and Olivia Redfield and Michael Collins and Ankur Parikh and Chris Alberti and Danielle Epstein and Illia Polosukhin and Matthew Kelcey and Jacob Devlin and Kenton Lee and Kristina N. Toutanova and Llion Jones and Ming-Wei Chang and Andrew Dai and Jakob Uszkoreit and Quoc Le and Slav Petrov},
year	= {2019},
journal	= {Transactions of the Association of Computational Linguistics}
}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class NQF1(datasets.Metric):
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
        language=datasets.Value("string"),  # Kept for schema compatibility (unused in NQF1)
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

        if not predictions:
            raise ValueError("No predictions provided")
        elif not references:
            raise ValueError("No references provided")

        predictions = dict(map(self._convert_pred_to_entry, predictions))
        references = dict(map(self._convert_ref_to_entry, references))

        # TODO: parameterize
        skip_missing_example_ids = False
        long_non_null_threshold = 2
        short_non_null_threshold = 2

        long_answer_stats, short_answer_stats = score_answers(
            gold_annotation_dict=references, pred_dict=predictions,
            skip_missing_example_ids=skip_missing_example_ids,
            long_non_null_threshold=long_non_null_threshold,
            short_non_null_threshold=short_non_null_threshold)

        metrics = pretty_print(long_answer_stats=long_answer_stats, short_answer_stats=short_answer_stats)
        return metrics

    def _convert_ref_to_entry(self, ref: dict) -> Tuple[str, List[NQLabel]]:
        """
        Converts a reference dict into an example_id, [labels] pair.
        """
        if not all(ref['example_id'][0] == ref['example_id'][i] for i in range(len(ref['example_id']))):
            raise ValueError("Found mismatched examples")
        elif not all(ref['language'][0] == ref['language'][i] for i in range(len(ref['language']))):
            raise ValueError("Found mismatched languages")

        key = ref['example_id'][0]
        value = [
            NQLabel(
                example_id=ref['example_id'][i],
                long_answer_span=self._passage_index_to_long_span(ref['passage_index'][i]),
                short_answer_span_list=[NQSpan(
                    start_byte=ref['start_position'][i],
                    end_byte=ref['end_position'][i],
                    start_token=-1,
                    end_token=-1)]
                ,
                yes_no_answer=self._bool_target(
                    TargetType(ref['yes_no_answer'][i])
                ),
                long_score=0,
                short_score=0,
            ) for i in range(len(ref['passage_index']))
        ]
        return key, value

    def _convert_pred_to_entry(self, pred: dict) -> Tuple[str, NQLabel]:
        """
        Converts a prediction dict into an example_id, label pair.
        """
        key = pred['example_id']
        value = NQLabel(
            example_id=pred['example_id'],
            long_answer_span=self._passage_index_to_long_span(pred['passage_index']),
            short_answer_span_list=[NQSpan(
                pred['start_position'],
                pred['end_position'],
                start_token=-1,
                end_token=-1)]
            ,
            yes_no_answer=self._bool_target(
                TargetType(pred['yes_no_answer'])
            ),
            long_score=pred['confidence_score'],
            short_score=pred['confidence_score'],
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

    @staticmethod
    def _passage_index_to_long_span(passage_index: int) -> NQSpan:
        if passage_index == -1:
            return NQSpan.null_span()
        else:
            return NQSpan(
                start_byte=passage_index,
                end_byte=passage_index,
                start_token=-1,
                end_token=-1
            )


