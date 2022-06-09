from datasets import Dataset
import pytest
from pytest import raises
import json

import numpy as np

from oneqa.boolqa.processors.postprocessors.boolqa_classifier import BoolQAClassifierPostProcessor

from tests.oneqa.mrc.common.base import UnitTest


class TestExtractivePostProcessor(UnitTest):
    _predict_scores=np.array([[-4.8621945,  7.0225224, -1.3763661],
       [ 5.729403 , -1.9911858, -3.4178042],
       [-4.871264 ,  2.7192504,  2.9824948]])
    _expected_predictions0=np.array([1, 0, 2])
    _expected_predictions1=np.array([2, 0, 2])

    _examples = Dataset.from_json('tests/resources/boolqa/processors/postprocessors/examples.json')
    _features = Dataset.from_json('tests/resources/boolqa/processors/postprocessors/features.json')
    _expected_predictions=json.load(open('tests/resources/boolqa/processors/postprocessors/expected_predictions.json'))    
    _expected_processed_predictions=json.load(open('tests/resources/boolqa/processors/postprocessors/expected_processed_predictions.json'))
    #------------------------------------------------------
    # sample data obtained with this embedded in process_references_and_predictions()
    # # Dataset.from_dict(examples[[0,5,-2]]).to_json('examples.json')
    # # Dataset.from_dict(features[[0,5,-2]]).to_json('features.json')
    #------------------------------------------------------


    def test_get_prediction_from_predict_scores_no_drop(self):
        postprocessor_class = BoolQAClassifierPostProcessor
        postprocessor = postprocessor_class(
            k=10, 
            drop_label=None,
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )        
        predictions=postprocessor._get_prediction_from_predict_scores(self._predict_scores)
        assert(np.all(predictions==self._expected_predictions0))

    def test_get_prediction_from_predict_scores_drop(self):
        postprocessor_class = BoolQAClassifierPostProcessor 
        postprocessor = postprocessor_class(
            k=10, 
            drop_label='NONE',
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )        
        predictions=postprocessor._get_prediction_from_predict_scores(self._predict_scores)
        assert(np.all(predictions==self._expected_predictions1))        


    def test_process_references_and_predictions(self):
        postprocessor_class = BoolQAClassifierPostProcessor
        postprocessor = postprocessor_class(
            k=10, 
            drop_label='NONE',
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )                    
        eval_predictions_with_processing=postprocessor.process_references_and_predictions(self._examples, self._features, self._predict_scores)
        assert(self._expected_predictions==eval_predictions_with_processing.predictions)
        assert(self._expected_processed_predictions==eval_predictions_with_processing.processed_predictions)
