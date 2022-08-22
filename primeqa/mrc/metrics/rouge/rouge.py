import datasets
import numpy as np
from rouge import Rouge

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
Calculates average rouge scores for a list of hypotheses and references
Args:
    predictions: list of predictions to score. Each prediction
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    rougeL: rouge_l (precision, recall, f1),
Examples:
    >>> rouge = datasets.load_metric('rouge')
    >>> predictions = ["hello there", "general kenobi"]
    >>> references = ["hello there", "general kenobi"]
    >>> results = rouge.compute(predictions=predictions, references=references)
    >>> print(list(results.keys()))
    ['rougeL']
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class ROUGE(datasets.Metric):
    def _info(self):
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
        rouge = Rouge()
        # no normalization
        try:
            scores = rouge.get_scores(prediction, ground_truth, avg=True)
        except ValueError:  # "Hypothesis is empty."
            return 0.0
        return scores["rouge-l"]["f"]

    def _metric_max_over_ground_truths(self,metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    
    def _compute(self, predictions, references, **kwargs):
        # adopted KILT standard evaluation from 
        # https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py
        total_count = 0
        rougel = 0

        for pred,ref in zip(predictions,references):
            _id = pred["id"]
            _pred = pred["prediction_text"]
            assert ref["id"] == _id
            total_count += 1
            _refs = ref["answers"]
            local_rougel = self._metric_max_over_ground_truths(self._rougel_score, _pred, _refs)
            rougel += local_rougel
            
        result = {"rougeL": (rougel/total_count)*100}
        prediction_lens = [pred["prediction_text"].count(' ') for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result