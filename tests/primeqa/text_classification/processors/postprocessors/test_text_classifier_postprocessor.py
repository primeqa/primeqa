from datasets import Dataset
import pytest
from pytest import raises
import json

import numpy as np

from primeqa.text_classification.processors.postprocessors.text_classifier import TextClassifierPostProcessor
#from primeqa.boolqa.processors.postprocessors.boolqa_classifier import BoolQAClassifierPostProcessor

from tests.primeqa.mrc.common.base import UnitTest


class TestExtractivePostProcessor(UnitTest):
    _predict_scores_for_get_pred=np.array([[-4.8621945,  7.0225224, -1.3763661],
       [ 5.729403 , -1.9911858, -3.4178042],
       [-4.871264 ,  2.7192504,  2.9824948]])

    # we use rounded scores here to not trip on float comparison - change the reference files by hand!
    _predict_scores_for_process=np.array([[ -4.0, -3.4, 7.7 ] ,
                                     [ -4.3, -2.9, 7.4 ]])    

    _expected_predictions0=np.array([1, 0, 2])
    _expected_predictions1=np.array([2, 0, 2])

    _examples = Dataset.from_json('tests/resources/text_classification/processors/postprocessors/examples.json')
    _features = Dataset.from_json('tests/resources/text_classification/processors/postprocessors/features.json')
    _expected_references=json.load(open('tests/resources/text_classification/processors/postprocessors/references.json'))
    _expected_predictions=json.load(open('tests/resources/text_classification/processors/postprocessors/predictions.json'))
    _expected_processed_predictions=json.load(open('tests/resources/text_classification/processors/postprocessors/processed_predictions.json'))
    _expected_processed_predictions_no_ref=json.load(open('tests/resources/text_classification/processors/postprocessors/processed_predictions_no_ref.json'))
    #------------------------------------------------------
    # sample data obtained with this embedded in process_references_and_predictions()
    # be sure to round scores by hand to avoid floating point comparison issues
    #        Dataset.from_dict(examples[[9, 101]]).to_json('examples.json', force_ascii=False)
    #        Dataset.from_dict(features[[9,101]]).to_json('features.json', force_ascii=False)    
    #...
    # r=[references[9], references[101]]
    # p={ '1387864999024164096': examples_json['1387864999024164096'],'-8692093041318212608': examples_json['-8692093041318212608'] } 
    # pp=[preds_for_metric[9], preds_for_metric[101]]
    # import json
    # json.dump(r, open('references.json','wt'))
    # json.dump(p, open('predictions.json','wt'))
    # json.dump(pp, open('process_predictions.json','wt'))
    # 
    #------------------------------------------------------


    def test_get_prediction_from_predict_scores_no_drop(self):
        postprocessor_class = TextClassifierPostProcessor
        postprocessor = postprocessor_class(
            k=10, 
            drop_label=None,
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )        
        predictions=postprocessor._get_prediction_from_predict_scores(self._predict_scores_for_get_pred)
        assert(np.all(predictions==self._expected_predictions0))

    def test_get_prediction_from_predict_scores_drop(self):
        postprocessor_class = TextClassifierPostProcessor
        postprocessor = postprocessor_class(
            k=10, 
            drop_label='NONE',
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )        
        predictions=postprocessor._get_prediction_from_predict_scores(self._predict_scores_for_get_pred)
        assert(np.all(predictions==self._expected_predictions1))        


    def test_process_references_and_predictions(self): 
        postprocessor_class = TextClassifierPostProcessor
        postprocessor = postprocessor_class(
            k=10, 
            drop_label='NONE',
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )                    
        eval_predictions_with_processing=postprocessor.process_references_and_predictions(self._examples, self._features, self._predict_scores_for_process)
        assert(self._expected_predictions==eval_predictions_with_processing.predictions)
        assert(self._expected_processed_predictions==eval_predictions_with_processing.processed_predictions)
        assert(self._expected_references==eval_predictions_with_processing.label_ids)

    def test_process(self): 
        postprocessor_class = TextClassifierPostProcessor
        postprocessor = postprocessor_class(
            k=10, 
            drop_label='NONE',
            label_list =['False', 'NONE', 'True'],
            id_key="example_id",
            output_label_prefix="evc"
        )                    
        eval_predictions_with_processing=postprocessor.process(self._examples, self._features, self._predict_scores_for_process)
        assert(self._expected_processed_predictions_no_ref==eval_predictions_with_processing.processed_predictions)
