from transformers import AutoTokenizer
from primeqa.mrc.data_models.target_type import TargetType
from primeqa.boolqa.processors.preprocessors.boolqa_classifier import BoolQAClassifierPreProcessor
from primeqa.boolqa.processors.dataset.mrc2dataset  import create_dataset_from_run_mrc_output
from tests.primeqa.mrc.common.base import UnitTest



class TestNWayPreProcessor(UnitTest):
    _eval_pred_in_file = 'tests/resources/boolqa/processors/preprocessors/eval_predictions.json'
    _expected_eval_dataset0 = {'example_id': '9d854612-d484-4b98-b208-4983baeeab67',
                              'language': 'english',
                              'question': 'Do zebra finches have stripes?',
                              'label': 0,
                              'span_answer_text': 'There are two subspecies of the zebra finch:',
                              'input_ids': [0, 984, 116232, 2270, 17007, 765, 43613, 90, 32, 2, 2, 8622, 621, 6626, 1614, 16711, 3387, 111, 70, 116232, 2270, 206, 12, 2],
                              'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
                              }
    _evc_model_path='/dccstor/jsmc-nmt-01/bool/git/IOTA-boolean-challenge/model/evc-c5'
    


    def test_load_and_preprocess(self):
        preprocessor_class = BoolQAClassifierPreProcessor
        tokenizer=AutoTokenizer.from_pretrained(self._evc_model_path, use_fast=True)
        preprocessor = preprocessor_class(
            sentence1_key='question',
            sentence2_key='span_answer_text',
            tokenizer=tokenizer,
            load_from_cache_file=False,
            max_seq_len=500,
            padding=False
        ) 
        examples=create_dataset_from_run_mrc_output(self._eval_pred_in_file, unpack=False)
        eval_examples, eval_dataset = preprocessor.process_eval(examples)
        print(eval_dataset[0])
        print(self._expected_eval_dataset0)
        assert(eval_dataset[0] == self._expected_eval_dataset0)