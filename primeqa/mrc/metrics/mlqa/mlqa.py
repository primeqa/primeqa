""" MLQA metric. """

import datasets
from typing import Dict, Any
from primeqa.mrc.metrics.mlqa.mlqa_evaluation_v1 import evaluate


_CITATION = """\
@article{lewis2019mlqa,
  title={MLQA: Evaluating Cross-lingual Extractive Question Answering},
  author={Lewis, Patrick and Oguz, Barlas and Rinott, Ruty and Riedel, Sebastian and Schwenk, Holger},
  journal={arXiv preprint arXiv:1910.07475},
  year={2019}
}
"""

_DESCRIPTION = """
This metric wrap the official scoring script for version 1 of the MultiLingual Question Answering (MLQA).


MLQA (MultiLingual Question Answering) is a benchmark dataset for evaluating cross-lingual question 
answering performance. MLQA consists of over 5K extractive QA instances (12K in English) in SQuAD 
format in seven languages - English, Arabic, German, Spanish, Hindi, Vietnamese and Simplified Chinese. 
MLQA is highly parallel, with QA instances parallel between 4 different languages on average.
"""

_KWARGS_DESCRIPTION = """
Computes MLQA SQuAD scores (F1 and EM).
Args:
    predictions: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair as given in the references (see below)
        - 'prediction_text': the text of the answer
    references: List of question-answers dictionaries with the following key-values:
        - 'id': id of the question-answer pair (see above),
        - 'answers': a Dict in the SQuAD dataset format
            {
                'text': list of possible texts for the answer, as a list of strings
                'answer_start': list of start positions for the answer, as a list of ints
            },
        - 'answer_language' the language of the answer
            Note that answer_start values are not taken into account to compute the metric.
Returns:
    'exact_match': Exact match (the normalized answer exactly match the gold answer)
    'f1': The F-score of predicted tokens versus the gold answer
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class MLQA(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"id": datasets.Value("string"), "prediction_text": datasets.Value("string")},
                    "references": {
                        "id": datasets.Value("string"),
                        "answers": datasets.features.Sequence(
                            {
                                "text": datasets.Value("string"),
                                "answer_start": datasets.Value("int32"),
                            }
                        )
                    },
                }
            ),
            codebase_urls=["https://github.com/facebookresearch/MLQA"],
            reference_urls=["https://github.com/facebookresearch/MLQA"],
        )

    def _compute(self, *, predictions, references, **kwargs) -> Dict[str, Any]:     
        
        if not predictions:
                raise ValueError("No predictions provided")
        elif not references:
            raise ValueError("No references provided")
        if not kwargs or 'dataset_config_name' not in  kwargs:
            raise ValueError("No dataset config name provided")
        
        pred_dict = {prediction["id"]: prediction["prediction_text"] for prediction in predictions}
        dataset = [
            {
                "paragraphs": [
                    {
                        "qas": [
                            {
                                "answers": [{"text": answer_text} for answer_text in ref["answers"]["text"]],
                                "id": ref["id"],
                            }
                            for ref in references
                        ]
                    }
                ]
            }
        ]
        answer_language = kwargs['dataset_config_name'][5:7]
        score = evaluate(dataset=dataset, predictions=pred_dict, lang=answer_language)
        return score