import datasets
import pytest
from pytest import raises

from primeqa.mrc.processors.postprocessors.mlqa import MLQAPostProcessor
from primeqa.mrc.processors.preprocessors.mlqa import MLQAPreprocessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from tests.primeqa.mrc.common.base import UnitTest


class TestMLQAPostProcessor(UnitTest):
    
    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def mlqa_eval_examples(self):
        examples = datasets.load_dataset("mlqa", "mlqa.zh.en", split='validation[:5]')
        return examples
        
    @pytest.fixture(scope='session')
    def mlqa_preprocessor(self, tokenizer):
        return MLQAPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            max_q_char_len=128
        )
        
    @pytest.fixture(scope='session')
    def mlqa_eval_examples_and_features(self, mlqa_eval_examples, mlqa_preprocessor):
        return mlqa_preprocessor.process_eval(mlqa_eval_examples)
   
    def test_post_processor_has_examples_and_features(self, mlqa_eval_examples_and_features):
        eval_examples, _ = mlqa_eval_examples_and_features
        postprocessor_class = MLQAPostProcessor
        scorer_type='weighted_sum_target_type_and_score_diff'
        postprocessor = postprocessor_class(k=20, n_best_size=20, max_answer_length=32,
                                            scorer_type=SupportedSpanScorers(scorer_type),
                                            single_context_multiple_passages=True)
        references = postprocessor.prepare_examples_as_references(eval_examples)
        for reference in references:
            assert reference['answer_language'] == 'zh'
            
                    




    
