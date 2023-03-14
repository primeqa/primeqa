import datasets
import pytest
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer

from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from tests.primeqa.mrc.common.base import UnitTest


class TestSQUADQAPreprocessor(UnitTest):

    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def train_examples(self):
        examples = datasets.load_dataset("squad", "plain_text", split='train[:100]')
        return examples

    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def eval_examples(self):
        examples = datasets.load_dataset("squad", "plain_text", split='validation[:100]')
        return examples
    
    @pytest.fixture(scope='session')
    def original_squad_examples(self):
        examples = {
            "data": [ [
            {
                "title": "Super_Bowl_50",
                "paragraphs": [
                    {
                        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24\u201310 to earn their third Super Bowl title. The game was played on February 7, 2016, at Levi's Stadium in the San Francisco Bay Area at Santa Clara, California. As this was the 50th Super Bowl, the league emphasized the \"golden anniversary\" with various gold-themed initiatives, as well as temporarily suspending the tradition of naming each Super Bowl game with Roman numerals (under which the game would have been known as \"Super Bowl L\"), so that the logo could prominently feature the Arabic numerals 50.",
                        "qas": [
                            {
                                "answers": [
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                },
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                },
                                {
                                    "answer_start": 177,
                                    "text": "Denver Broncos"
                                }
                            ],
                            "question": "Which NFL team represented the AFC at Super Bowl 50?",
                            "id": "56be4db0acb8001400a502ec"
                            },
                        ]
                    }
                ]
            }
            ]
        ],
        "version": "1.1"
        }
        return examples

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def squad_preprocessor(self, tokenizer):
        return SQUADPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
        )

    def test_train_preprocessing_runs_without_errors(self, train_examples, squad_preprocessor):
        train_examples, train_features = squad_preprocessor.process_train(train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        for example in train_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, squad_preprocessor):
        eval_examples, eval_features = squad_preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        for example in eval_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
            
    def test_original_squad_format_runs_without_errors(self, original_squad_examples, squad_preprocessor):
        raw_dataset = datasets.Dataset.from_pandas(pd.DataFrame(data=original_squad_examples))
        examples,features = squad_preprocessor.process_eval(raw_dataset)
        assert isinstance(examples, Dataset)
        assert isinstance(features, Dataset)
        for example in examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
