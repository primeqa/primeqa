from collections import defaultdict
from itertools import groupby
from operator import itemgetter

from tqdm import tqdm

from oneqa.mrc.processors.postprocessors.abstract import AbstractPostProcessor


class ExtractivePostProcessor(AbstractPostProcessor):
    def __init__(self, k: int, n_best_size: int, max_answer_length: int, span_tracker_factory):
        super().__init__(k)
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length
        self._span_trackers = defaultdict(span_tracker_factory)  # TODO factory type?

    def process(self, examples, features, predictions):
        features_itr = groupby(features, key=itemgetter('example_id'))
        predictions_i = 0
        if len(features) != predictions[0].shape[0] and all(
                p.shape[0] == predictions[0].shape[0] for p in predictions[1:]):
            raise ValueError(f"Size mismatch withing {len(features)} features and predictions "
                             f"of first dim {[p.shape[0] for p in predictions]}")

        for example in tqdm(examples):
            example_id, example_features = next(features_itr)
            if example_id != example['example_id']:
                raise ValueError(f"Example id mismatch between example ({example['example_id']}) "
                                 f"and feature ({example_id})")
            example_features = list(example_features)
            example_predictions = None  # [i, i + len(example_features)


