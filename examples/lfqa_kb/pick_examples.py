# Find examples that have good retrieved passages but bad bart generation.
from rouge import Rouge
import numpy as np
import json

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
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
    max_idx = np.argmax(scores_for_ground_truths)
    return scores_for_ground_truths[max_idx], ground_truths[max_idx] 

import argparse
import os

parser=argparse.ArgumentParser()
parser.add_argument('--output_dir', help='for example, ""/dccstor/myu/experiments/eli5_examples/""')
parser.add_argument('--data_fn', help='the Eli5 dev data file with DPR retrieved passages')
parser.add_argument('--pred_fn', help='the dev prediction file of a generation model (i.e., BART)')
args=parser.parse_args()


output_dir = args.output_dir
data_fn = args.data_fn
pred_fn = args.pred_fn
# output_dir = "/dccstor/myu/experiments/eli5_examples/"
# data_fn = "/dccstor/myu/data/kilt_eli5_dpr/eli5-dev-kilt-dpr.json"
# pred_fn = "/dccstor/myu/experiments/eli5_bart_large_beam_lr3e-5/eval_predictions.json"
assert os.path.isfile(data_fn)
assert os.path.isfile(pred_fn)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
assert os.path.isdir(output_dir)


# 1. take the concatenation of the dpr top-3 retrieved passages as prediction
rouge_scores = []
max_score_refs = []
predictions = []
references = []

with open(data_fn, 'r', encoding='utf-8') as f, open(output_dir + "psg_top3_rougel.tsv", 'w') as fw:
    fw.write("score\tmax_score_ref\tprediction\n")
    for line in f:
        data = json.loads(line.strip())
        _refs = [x["answer"] for x in data["output"] if "answer" in x]
        references.append(_refs)
        _pred = " ".join([x["text"] for x in data["passages"][:3]])

        local_rougel, max_score_ref = _metric_max_over_ground_truths(
            _rougel_score, _pred, _refs
        )
        pred_line = _pred.replace('\n', ' ')
        fw.write(f"{local_rougel:.4f}\t{max_score_ref}\t{pred_line}\n")
    print("saved top3 passages scores.")


# 2. take each of the top-3 passages as prediction
rouge_scores = []
max_score_refs = []
predictions = []
references = []

with open(data_fn, 'r', encoding='utf-8') as f, open(output_dir + "psg_best_rougel.tsv", 'w') as fw:
    fw.write("score\tmax_score_ref\tprediction\n")
    for line in f:
        data = json.loads(line.strip())
        _refs = [x["answer"] for x in data["output"] if "answer" in x]
        references.append(_refs)
        max_local_score_ref = ""
        max_local_rouge = 0.0
        best_psg = ""
        for psg in [x["text"] for x in data["passages"][:3]]:
            _pred = psg
            local_rougel, max_score_ref = _metric_max_over_ground_truths(
                _rougel_score, _pred, _refs
            )
            if local_rougel > max_local_rouge:
                max_local_rouge = local_rougel
                max_local_score_ref = max_score_ref
                best_psg = psg
        pred_line = _pred.replace('\n', ' ')
        fw.write(f"{max_local_rouge:.4f}\t{max_local_score_ref}\t{pred_line}\n")
    print("saved the best passage scores.")


# 3. take the bart generation as prediction
with open(pred_fn, 'r') as f:
    predictions = json.loads(f.read())

rouge_scores = []
max_score_refs = []
for idx,line in enumerate(predictions):

    _refs = references[idx]
    _pred = line["prediction_text"]

    local_rougel, max_score_ref = _metric_max_over_ground_truths(
        _rougel_score, _pred, _refs
    )
    rouge_scores.append(local_rougel)
    max_score_refs.append(max_score_ref)
with open(output_dir + "bart_rougel.tsv", 'w') as f:
    f.write("score\tmax_score_ref\tprediction\n")
    for idx,line in enumerate(rouge_scores):
        pred_line = predictions[idx]['prediction_text'].replace('\n', ' ')
        f.write(f"{line:.4f}\t{max_score_refs[idx]}\t{pred_line}\n")
    print("saved bart generation scores.")


# Comparison: top3 psgs v.s. bart
print("Comparison: Top3 Passages v.s. BART")
with open(output_dir + "psg_top3_rougel.tsv") as f:
    dpr_rouges = [line.strip().split('\t')[0] for line in f][1:] # ignore header
    print(dpr_rouges[:10])
with open(output_dir + "bart_rougel.tsv") as f:
    bart_rouges = [line.strip().split('\t')[0] for line in f][1:] # ignore header
    print(bart_rouges[:10])
with open(output_dir + "bart_rougel.tsv") as f:
    predictions = [line.strip().split('\t')[2] for line in f][1:] # ignore header

gains = [float(dpr_rouges[i]) - float(bart_rouges[i]) for i in range(len(dpr_rouges))]
gain_ordered_index = np.argsort(gains)[::-1]

with open(data_fn, 'r') as f:
    dev_data = [line.strip() for line in f]

max_gain_preds = [predictions[x] for x in gain_ordered_index[:] if gains[x] > 0]
max_gain_examples = [dev_data[x] for x in gain_ordered_index[:] if gains[x] > 0]
max_gains = [gains[x] for x in gain_ordered_index[:] if gains[x] > 0]


print("There are", len(max_gains), "examples that have ROUGE-L gain")

with open(output_dir+"max_rouge_gain_dev_examples_top3psgs-bart.json", 'w') as f:
    for idx,line in enumerate(max_gain_examples):
        data = json.loads(line.strip())
        data["bart_prediction"] = max_gain_preds[idx]
        data["passages"] = data["passages"][-3:]
        data["rouge_gain"] = gains[idx]
        
        f.write(json.dumps(data) + '\n')


# Comparison: top3 psgs v.s. bart
print("Comparison: Best Passage v.s. BART")
with open(output_dir + "psg_best_rougel.tsv") as f:
    dpr_rouges = [line.strip().split('\t')[0] for line in f][1:] # ignore header
    print(dpr_rouges[:10])
with open(output_dir + "bart_rougel.tsv") as f:
    bart_rouges = [line.strip().split('\t')[0] for line in f][1:] # ignore header
    print(bart_rouges[:10])
with open(output_dir + "bart_rougel.tsv") as f:
    predictions = [line.strip().split('\t')[2] for line in f][1:] # ignore header

gains = [float(dpr_rouges[i]) - float(bart_rouges[i]) for i in range(len(dpr_rouges))]
gain_ordered_index = np.argsort(gains)[::-1]

with open(data_fn, 'r') as f:
    dev_data = [line.strip() for line in f]

max_gain_preds = [predictions[x] for x in gain_ordered_index[:] if gains[x] > 0]
max_gain_examples = [dev_data[x] for x in gain_ordered_index[:] if gains[x] > 0]
max_gains = [gains[x] for x in gain_ordered_index[:] if gains[x] > 0]


print("There are", len(max_gains), "examples that have ROUGE-L gain")

with open(output_dir+"max_rouge_gain_dev_examples_bestpsgs-bart.json", 'w') as f:
    for idx,line in enumerate(max_gain_examples):
        data = json.loads(line.strip())
        data["bart_prediction"] = max_gain_preds[idx]
        data["passages"] = data["passages"][-3:]
        data["rouge_gain"] = gains[idx]
        
        f.write(json.dumps(data) + '\n')