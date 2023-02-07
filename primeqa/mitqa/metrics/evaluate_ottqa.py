import json
import re
import collections
import string
import sys
from primeqa.mitqa.metrics.evaluate import normalize_answer,get_tokens,compute_exact,compute_f1

def get_raw_scores(examples, reference):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    exact_scores = {}
    f1_scores = {}
    
    for example in examples:
        qas_id = example['question_id']
        gold_answers = [reference['reference'][qas_id]]

        prediction = example['pred']
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    qid_list = reference['reference'].keys()
    total = len(qid_list)
    
    for k in qid_list:
        if k not in exact_scores:
            print("WARNING: MISSING QUESTION {}".format(k))
    qid_list = list(set(qid_list) & set(exact_scores.keys()))

    return collections.OrderedDict(
        [
            ("total exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
            ("total f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
            ("total", total),
        ]
    )

    
def get_em_and_f1_ottqa(data_file,ref_file):
    data = json.load(open(data_file))
    ref = json.load(open(ref_file))
    return get_raw_scores(data, ref)
    

