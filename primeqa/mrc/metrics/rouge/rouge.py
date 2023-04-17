import datasets
import numpy as np
from rouge import Rouge
from rouge_score import rouge_scorer


_CITATION = """\
@inproceedings{lin-2004-rouge,
    title = "{ROUGE}: A Package for Automatic Evaluation of Summaries",
    author = "Lin, Chin-Yew",
    booktitle = "Text Summarization Branches Out",
    month = jul,
    year = "2004",
    address = "Barcelona, Spain",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W04-1013",
    pages = "74--81",
}
"""

_DESCRIPTION = """\
ROUGE, or Recall-Oriented Understudy for Gisting Evaluation, is a set of metrics and a software package used for
evaluating automatic summarization and machine translation software in natural language processing.
The metrics compare an automatically produced summary or translation against a reference or a set of references (human-produced) summary or translation.
Note that ROUGE is case insensitive, meaning that upper case letters are treated the same way as lower case letters.
This metrics is a wrapper around Google Research reimplementation of ROUGE:
https://github.com/google-research/google-research/tree/master/rouge
"""

_KWARGS_DESCRIPTION = """
Calculates average rougeL scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each prediction
        should contain an 'id' and a 
        'prediction_text': string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should contain an 'id' and an 
        'answers' list of string with tokens separated by spaces.
Returns:
    rougeL: rouge_l (precision, recall, f1),
Examples:
    >>> rouge = datasets.load_metric('rouge')
    >>> predictions = [{"id":1, "prediction_text" : "hello there"}, "{"id":2, "prediction_text": "general kenobi"]
    >>> references = [{"id":1, "answers" : ["hello there"]}, {"id":2, "answers": ["general kenobi"]}]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['rougeL', 'gen_len']
"""

import sys

@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ROUGE(datasets.Metric):
    
    def _info(self):
        self._hf_rouge = rouge_scorer.RougeScorer(rouge_types=['rougeLsum'], split_summaries=True)
        self._kilt_rouge = Rouge(metrics=['rouge-l'])
        sys.setrecursionlimit(20000)
        
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": {"id":datasets.Value("string", id = None),"prediction_text":datasets.Value(dtype='string', id=None)},
                    "references": {"id":datasets.Value("string", id = None),"answers":datasets.Sequence(datasets.Value(dtype='string', id=None))},
                }
            ),
            codebase_urls=["https://github.com/google-research/google-research/tree/master/rouge"],
            reference_urls=[
                "https://en.wikipedia.org/wiki/ROUGE_(metric)",
                "https://github.com/google-research/google-research/tree/master/rouge",
            ],
        )
        
    def _rougel_score(self,prediction, ground_truth):
        try:
            hf_scores = self._hf_rouge.score(ground_truth, prediction)
            kilt_scores = self._kilt_rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:  # "Hypothesis is empty."
            return 0.0, 0.0
        return hf_scores['rougeLsum'].fmeasure, kilt_scores["rouge-l"]["f"]

    def _metric_max_over_ground_truths(self, prediction, ground_truths):
        kilt_scores_for_ground_truths = []
        google_scores_for_ground_truths = []
        for ground_truth in ground_truths:
            hf_score, kilt_score = self._rougel_score(prediction, ground_truth)
            google_scores_for_ground_truths.append(hf_score)
            kilt_scores_for_ground_truths.append(kilt_score)
        return max(google_scores_for_ground_truths), max(kilt_scores_for_ground_truths)

    
    def _compute(self, predictions, references, **kwargs):
        # adopted KILT standard evaluation from 
        # https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py
        total_count = 0
        kilt_rougel = 0
        google_rougel = 0

        for pred,ref in zip(predictions,references):
            _id = pred["id"]
            _pred = pred["prediction_text"]
            assert ref["id"] == _id
            total_count += 1
            _refs = ref["answers"]
            google_local_rougel, kilt_local_rougel = self._metric_max_over_ground_truths(_pred, _refs)
            kilt_rougel += kilt_local_rougel
            google_rougel += google_local_rougel
            
        result = {"kilt_rougeL": (kilt_rougel/total_count)*100, "google_rougeL": (google_rougel/total_count)*100}
        prediction_lens = [pred["prediction_text"].count(' ') for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result