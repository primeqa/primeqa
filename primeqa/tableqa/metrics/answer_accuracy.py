import ast
import logging
logger = logging.getLogger(__name__)

def compute_denotation_accuracy(predicted_answers,gold_answers):
    """Computes denotation accuracy based on predicted and gold answers

    Args:
        predicted_answers (List): List of predicted answers
        gold_answers (List): List of gold answers

    Returns:
        Dict: Metrics score
    """
    
    exact_match = []
    for pred,gold in zip(predicted_answers,gold_answers):
        correct = 0
        if pred==gold:
            correct = 1
        exact_match.append(correct)
    accuracy = sum(exact_match) / len(exact_match)
    return {"Denotation accuracy":accuracy}