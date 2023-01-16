import json
import re
import collections
import string
import sys

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """
    The get_tokens function takes a string as input and returns the tokens in that string.
    The function normalizes the answer to lowercase, removes punctuation, and splits on whitespace.
    
    
    Args:
        s: Normalize the string
    
    Returns:
        A list of the normalized tokens in a string
    """
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    """
    The compute_exact function takes in two arguments, a_gold and a_pred.
    The function normalizes the gold answer and the predicted answer (using the normalize_answer function)
    and checks whether they are equal. If so, it returns 1; otherwise 0.
    
    Args:
        a_gold: Store the gold answer and a_pred is used to store the predicted answer
        a_pred: Store the predicted answer
    
    Returns:
        An integer value of 1 if the normalized answers match exactly, and 0 otherwise
    
    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    """
    The compute_f1 function takes in two arguments: a_gold and a_pred.
    The 'a_gold' argument is the list of words that are found in the gold
    standard, and 'a_pred' is the list of words that our system outputs as 
    the answer. The compute f function then calculates whether any of these 
    words match, and if so how many there are. If either input has length 0, then F=0.
    
    Args:
        a_gold: Pass in the gold answer
        a_pred: Store the predicted answer from the model
    
    Returns:
        The f-score for the two arguments
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

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
    

