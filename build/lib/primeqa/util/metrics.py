import logging
from sklearn.metrics import f1_score
import sklearn.metrics

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def multiclass_score_metrics(scores, preds, labels, *, average='micro'):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds, average=average)
    return {
        "acc": acc,
        "f1": f1
    }


def score_metrics(scores, preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = f1_score(y_true=labels, y_pred=preds)
    roc = sklearn.metrics.roc_auc_score(labels == 1, scores)
    return {
        "acc": acc,
        "roc": roc,
        "f1": f1,
        "positive_fraction": labels.mean()
    }
