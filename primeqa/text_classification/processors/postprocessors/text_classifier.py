from typing import List, Dict, Any

from datasets import Dataset
import numpy as np
import logging
from transformers import EvalPrediction

from primeqa.mrc.processors.postprocessors.abstract import AbstractPostProcessor
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing


logger = logging.getLogger(__name__)


class TextClassifierPostProcessor(AbstractPostProcessor):
    """
    Post processor for use with the text classifier
    """    
    def __init__(self, 
                drop_label: str,
                id_key: str,
                label_list: List[str],
                output_label_prefix: str,
                *args, **kwargs):
        """
        Args:
            drop_label: if specified, ignore this category for classifier output,
                     e.g. "no_answer" converts a ("yes","no_answer","no") classifier into a ("yes", "no") classifier
            id_key: unique identifier field of examples to be classified
            label_list: the (human-readable) labels produced by the classifier
            output_label_prefix: prefix for new output fields in eval_predictions.json
            *args: Arguments for super class constructor.
            **kwargs: Keyword Arguments for super class constructor.
        """                
        super().__init__(1, 1)
        self.id_key=id_key
        self.drop_label = drop_label
        self.label_list = label_list
        self.output_label_prefix = output_label_prefix


    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        """
        Convert examples into references for use with metrics.
        """
        references = []
        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            label = {
                'label': example['label'],
                'example_id': example['example_id'],
                'language': example['language'] if 'language' in example else 'default',
                'question': example['question']
            }
            references.append(label)
        return references


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



    def process(self, examples, features, predict_scores) -> EvalPrediction:
        print('in process')
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
                "language":ex["language"] if "language" in ex else "default",
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



    def process_references_and_predictions(self, examples, features, predict_scores) -> EvalPrediction:
        print('in process_references_and_predictions')
        references = self.prepare_examples_as_references(examples)
        ipredictions=self._get_prediction_from_predict_scores(predict_scores)

        fields = zip(features[self.id_key],
            ipredictions, 
            predict_scores,
            references)

        preds_for_metric=[]
        examples_json={}
        for (example_id, item, scores, example) in fields:
            item_label = self.label_list[item]

            p = {
                "prediction":str(item_label),
                "confidence":str(scores[item]),
                "example_id":str(example_id),
                "question":example["question"],
                "language":example["language"] if "language" in example else "default",
                "scores":{ label:float(score) for label,score in zip(self.label_list, scores)}
            }
            preds_for_metric.append(p)

            pred_ex = {}
            pred_ex['example_id'] = example_id
            pred_ex['prediction'] = p['prediction']
            pred_ex['confidence'] = p['confidence']
            pred_ex['question'] = p['question']
            pred_ex['language'] = p['language']
            pred_ex['scores'] = p['scores']
            examples_json[example_id] = [ pred_ex ]

        # noinspection PyTypeChecker
        return EvalPredictionWithProcessing(
            label_ids=references,
            predictions=examples_json,
            processed_predictions=preds_for_metric,
        )    
