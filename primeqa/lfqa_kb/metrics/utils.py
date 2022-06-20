from rouge import Rouge
from transformers.trainer_utils import EvalPrediction
import numpy as np

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def _metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


# modified from 
# https://github.com/huggingface/transformers/blob/198c335d219a5eb4d3f124fdd1ce1a9cd9f78a9b/examples/pytorch/summarization/run_summarization.py#L563
def compute_metrics(p: EvalPrediction):
    # adopted KILT standard evaluation from https://github.com/facebookresearch/KILT/blob/main/kilt/eval_downstream.py
    total_count = 0
    rougel = 0
    preds =p.predictions
    refs = p.label_ids

    for pred,ref in zip(preds,refs):
        _id = pred["id"]
        _pred = pred["prediction_text"]
        assert ref["id"] == _id
        total_count += 1
        _refs = ref["answers"]
        local_rougel = _metric_max_over_ground_truths(
            _rougel_score, _pred, _refs
        )
        rougel += local_rougel
    # result = metric.compute(predictions=preds, references=refs)
    # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    result = {"rougeL": (rougel/total_count)*100}
    prediction_lens = [pred["prediction_text"].count(' ') for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

