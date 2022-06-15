from typing import List, Dict, Any

from datasets import Dataset
import numpy as np
import logging
from transformers import EvalPrediction

from primeqa.mrc.processors.postprocessors.abstract import AbstractPostProcessor
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing


logger = logging.getLogger(__name__)


class BoolQAClassifierPostProcessor(AbstractPostProcessor):
    def __init__(self, 
                drop_label: str,
                id_key: str,
                label_list: List[str],
                output_label_prefix: str,
                *args, **kwargs):
        super().__init__(1, 1)
        self.id_key=id_key
        self.drop_label = drop_label
        self.label_list = label_list
        self.output_label_prefix = output_label_prefix


    def process(self, examples: Dataset, features: Dataset, predictions: tuple):
        """
        Convert data and model predictions into MRC answers.
        """
        pass

    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        """
        Convert examples into references for use with metrics.
        """
        pass

    def _get_prediction_from_predict_scores(self, predict_scores):
        if self.drop_label:
            # dropping NONE to get binary predictions from 3-way classifier
            # TODO maybe this should be model dependent rather than task dependent
            label_list=np.array(self.label_list)
            mask=label_list==self.drop_label
            masked_predict_scores=predict_scores.copy()
            masked_predict_scores[:,mask]=-9e19
        else:
            masked_predict_scores=predict_scores

        predictions = np.argmax(masked_predict_scores, axis=1)
        return predictions



    # TODO we aren't handling reference, metrics yet
    def process_references_and_predictions(self, examples, features, predict_scores) -> EvalPrediction:
        print('in process_references_and_predictions')
#        references = self.prepare_examples_as_references(examples)
        ipredictions=self._get_prediction_from_predict_scores(predict_scores)

        fields = zip(features[self.id_key],
            features["question"],
            ipredictions, 
            predict_scores)

        preds_for_metric=[]
        examples_json={}
        for (ex, (example_id, question, item, scores)) in zip(examples, fields):
            item_label = self.label_list[item]
            p = {
                "pred":str(item_label),
                "conf":str(scores[item]),
                "question":question,
                "language":ex["language"],
                "scores": { label:float(score) for label,score in zip(self.label_list, scores)}
            }
            preds_for_metric.append(p)

            ex[self.output_label_prefix+'_pred'] = p['pred']
            ex[self.output_label_prefix+'_scores'] = p['scores']
            ex[self.output_label_prefix+'_conf'] = p['conf']
            examples_json[example_id] = [ ex ]

        # noinspection PyTypeChecker
        return EvalPredictionWithProcessing(
            label_ids=None,
            predictions=examples_json,
            processed_predictions=preds_for_metric,
        )    
