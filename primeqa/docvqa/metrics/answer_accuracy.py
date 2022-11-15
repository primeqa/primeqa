import re
import ast
import string
import logging

from similarity.normalized_levenshtein import NormalizedLevenshtein


logger = logging.getLogger(__name__)

def normalize_answer(s, stop=False):
    """Applies basic string normalizations before comparison
    Args:
        s (str): Unnormalized string

    Returns:
        s (str): Normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|for|the|in|on|at|by|with)\b', ' ', text)
    def white_space_fix(text):
        return re.sub('\s\s*', ' ', text)
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    if stop:
        return white_space_fix(remove_articles(remove_punc(s))).strip()
    else:
        return white_space_fix(remove_punc(s)).strip()

def compute_anls_score(predicted_answers, gold_answers):
    """Computes average normalized Levenshtein similarity between predicted and gold answers

    Args:
        predicted_answers (List): List of predicted answers
        gold_answers (List): List of gold answers

    Returns:
        Dict: Metrics score
    """
    NLS = NormalizedLevenshtein()

    nls_score = 0
    for pred, golds in zip(predicted_answers, gold_answers):
        max_score = 0
        normalized_pred = normalize_answer(pred.lower(), stop=True)
        for gold in golds:
            normalized_gold = normalize_answer(gold.lower(), stop=True)
            similarity = NLS.similarity(normalized_gold, normalized_pred)
            similarity = similarity if similarity > 0.5 else 0.
            max_score = max(max_score, similarity)
        nls_score += max_score

    anls = 100 * nls_score / len(predicted_answers)
    return {"ANLS Score" : round(anls, 3)}
