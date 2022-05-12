from datasets import Dataset
import pytest
import sys
from pytest import raises
from transformers import AutoTokenizer
from itertools import groupby
from operator import itemgetter


import numpy as np

from oneqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from oneqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from oneqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from oneqa.mrc.data_models.target_type import TargetType
from tests.oneqa.mrc.common.base import UnitTest
from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES


class TestSQUADPostProcessor(UnitTest):
    
    @pytest.fixture(scope='session')
    def squad_eval_examples(self):
            question = ["Who killed Harold II?", "Who walked the dog?"]
            context = ["In 1066, Duke William II conquered England killing King Harold II. ",
                   "Bob walks the dog and Alice walks the cat."]
            example_id = ["sq-abc", "sq-123"]
            answer_start = [[9], [0]]
            text = [["Duke William II"], ["Bob"]]
            examples_dict = dict(question=question, context=context, id=example_id,
                             answers=[dict(text=t, answer_start=s)
                                     for t, s in
                                     zip(text, answer_start)])
            examples_dataset = Dataset.from_dict(examples_dict)
            return examples_dataset
        
    @pytest.fixture(scope='session')
    def squad_preprocessor(self, tokenizer):
        return SQUADPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            max_q_char_len=128
        )
        
    @pytest.fixture(scope='session')
    def squad_eval_examples_and_features(self, squad_eval_examples, squad_preprocessor):
        return squad_preprocessor.process_eval(squad_eval_examples)
        
    # Predictions obtained by ExtractivePostProcessor.process()
    predictions ={
        "sq-abc":[{"example_id": 'sq-abc',
                  "cls_score": -19.402141571044922,
                  "start_logit": 6.4368157386779785,
                  "end_logit": 6.1043925285339355,
                  "span_answer": {'start_position': 9, 'end_position': 24},
                  "span_answer_score": 17.68244457244873,
                  "start_index": 13,
                  "end_index": 15,
                  "passage_index": 0,
                  "target_type_logits": [-1.7478748559951782, 3.421539306640625, 1.0732213258743286, -1.5274697542190552, -1.6658639907836914],
                  "span_answer_text": 'Duke William II',
                  "yes_no_answer": 0,
                  "normalized_span_answer_score": 0.7221964058572311
                  }],
        "sq-123":[{"example_id": 'sq-123',
                  "cls_score": -19.262707710266113,
                  "start_logit": 5.269657611846924,
                  "end_logit": 5.499551773071289,
                  "span_answer": {'start_position': 0, 'end_position': 3},
                  "span_answer_score": 16.652730584144592,
                  "start_index": 8,
                  "end_index": 8,
                  "passage_index": 0,
                  "target_type_logits": [-1.7099565267562866, 3.2735440731048584, 1.0534870624542236, -1.5458885431289673, -1.6150139570236206],
                  "span_answer_text": 'Bob',
                  "yes_no_answer": 0,
                  "normalized_span_answer_score": 0.7932939080812123
                  }]
        }
    
    _expected_predictions = {
        'sq-abc' : {'prediction_text': "Duke William II"},
        'sq-123' : {'prediction_text': "Bob"}
    }
    
    _expected_references = {
        'sq-abc' : {'answers': {'text': ['Duke William II'], 'answer_start': [9]}},
        'sq-123' : {'answers': {'text': ['Bob'], 'answer_start': [0]}}
    }
   
    def test_post_processor_has_examples_and_features(self, squad_eval_examples_and_features):
        eval_examples, _ = squad_eval_examples_and_features
        postprocessor_class = SQUADPostProcessor
        scorer_type='weighted_sum_target_type_and_score_diff'
        postprocessor = postprocessor_class(k=20, n_best_size=20, max_answer_length=32,
                                            scorer_type=SupportedSpanScorers(scorer_type),
                                            single_context_multiple_passages=True)
        processed_predictions = postprocessor.prepare_predictions_for_squad(eval_examples, self.predictions)
        for predicted in  processed_predictions:
            expected = self._expected_predictions[predicted['id']]
            assert predicted['prediction_text'] == expected['prediction_text']
        references = postprocessor.prepare_examples_as_references(eval_examples)
        for reference in references:
            expected = self._expected_references[reference['id']]
            assert reference['answers']['text'] == expected['answers']['text']
            assert reference['answers']['answer_start'] == expected['answers']['answer_start']
            
                    




    
