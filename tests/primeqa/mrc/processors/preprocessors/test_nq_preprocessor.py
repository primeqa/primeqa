import datasets
from datasets import Dataset
from datasets.features.features import Sequence, Value, ClassLabel
import pytest
from pytest import raises
from transformers import AutoTokenizer
from primeqa.mrc.processors.preprocessors.natural_questions import NaturalQuestionsPreProcessor
from tests.primeqa.mrc.common.base import UnitTest


class TestNQPreprocessor(UnitTest):
    @pytest.fixture(scope='session')
    def nq_examples(self):
        features=datasets.Features(
            {
                "id": datasets.Value("string"),
                "document": {
                    "title": datasets.Value("string"),
                    "url": datasets.Value("string"),
                    "html": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(
                        {"is_html": datasets.Value("bool"), "token": datasets.Value("string"),
                        "start_byte": datasets.Value("int64"), "end_byte": datasets.Value("int64")}
                    ),
                },
                "question": {
                    "text": datasets.Value("string"),
                    "tokens": datasets.features.Sequence(datasets.Value("string")),
                },
                "long_answer_candidates": datasets.features.Sequence(
                    {
                        "start_token": datasets.Value("int64"),
                        "end_token": datasets.Value("int64"),
                        "start_byte": datasets.Value("int64"),
                        "end_byte": datasets.Value("int64"),
                        "top_level": datasets.Value("bool"),
                    }
                ),
                "annotations": datasets.features.Sequence(
                    {
                        "id": datasets.Value("string"),
                        "long_answer": {
                            "start_token": datasets.Value("int64"),
                            "end_token": datasets.Value("int64"),
                            "start_byte": datasets.Value("int64"),
                            "end_byte": datasets.Value("int64"),
                            "candidate_index": datasets.Value("int64")
                        },
                        "short_answers": datasets.features.Sequence(
                            {
                                "start_token": datasets.Value("int64"),
                                "end_token": datasets.Value("int64"),
                                "start_byte": datasets.Value("int64"),
                                "end_byte": datasets.Value("int64"),
                                "text": datasets.Value("string"),
                            }
                        ),
                        "yes_no_answer": datasets.features.ClassLabel(
                            names=["NO", "YES"]
                        ),  # Can also be -1 for NONE.
                    }
                )
            }
        )

        return Dataset.from_dict(
            {
                "id": ["000"],
                "document": [
                    {
                        "title": "Test NQ Preprocessor",
                        "url": "none",
                        "html": "<!DOCTYPE html> Alice walks the cat, Bob walks the dog",
                        "tokens": [
                            {"is_html": False, "token": "Alice", "start_byte": 16, "end_byte": 21},
                            {"is_html": False, "token": "walks", "start_byte": 22, "end_byte": 27},
                            {"is_html": False, "token": "the", "start_byte": 28, "end_byte": 31},
                            {"is_html": False, "token": "cat", "start_byte": 32, "end_byte": 35},
                            {"is_html": False, "token": ".", "start_byte": 35, "end_byte": 36},
                            {"is_html": False, "token": "Bob", "start_byte": 37, "end_byte": 40},
                            {"is_html": False, "token": "walks", "start_byte": 41, "end_byte": 46},
                            {"is_html": False, "token": "the", "start_byte": 47, "end_byte": 50},
                            {"is_html": False, "token": "dog", "start_byte": 51, "end_byte": 54}
                        ],
                    }
                ],
                "question": [
                    {
                        "text": "Who walked the dog?",
                        "tokens": ["Who", "walked", "the", "dog", "?"],
                    }
                ],
                "long_answer_candidates": [
                    {
                        "start_token": [0],
                        "end_token": [11],
                        "start_byte": [0],
                        "end_byte": [54],
                        "top_level": [True],
                    },
                ],
                "annotations": [
                    [
                        {
                            "id": "12345",
                            "long_answer": {
                                "start_token": 0,
                                "end_token": 11,
                                "start_byte": 0,
                                "end_byte": 54,
                                "candidate_index": 0,
                            },
                            "short_answers": [
                                {
                                    "start_token": 5,
                                    "end_token": 6,
                                    "start_byte": 37,
                                    "end_byte": 40,
                                    "text": "Bob",
                                },
                            ],
                            "yes_no_answer": -1,
                        }
                    ]
                ]
            },
            features
        )

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('roberta-base')
    
    @pytest.fixture(scope='class')
    def nq_preprocessor(self, tokenizer):
        return NaturalQuestionsPreProcessor(
                tokenizer,
                stride=128,
                load_from_cache_file=False,
            )

    def test_train_preprocessing_runs_without_errors(self, nq_examples, nq_preprocessor):
        train_examples, train_features = nq_preprocessor.process_train(nq_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        for example in train_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
            assert example['context_char_to_token'] is not None
            assert example['context_token_to_char'] is not None

    def test_eval_preprocessing_runs_without_errors(self, nq_examples, nq_preprocessor):
        eval_examples, eval_features = nq_preprocessor.process_eval(nq_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        for example in eval_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
