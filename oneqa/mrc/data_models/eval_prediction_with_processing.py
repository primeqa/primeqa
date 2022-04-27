from typing import NamedTuple, Tuple, Union

import numpy as np


class EvalPredictionWithProcessing(NamedTuple):
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Args:
        predictions: Predictions of the model.
        label_ids: Targets to be matched.
        processed_predictions: Predictions of the model processed for metric use.
    """

    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Union[np.ndarray, Tuple[np.ndarray]]
    processed_predictions: Union[np.ndarray, Tuple[np.ndarray]]