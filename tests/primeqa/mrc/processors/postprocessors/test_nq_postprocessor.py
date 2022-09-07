import datasets
from datasets import Dataset
import pytest
from transformers import AutoTokenizer
import torch
from primeqa.mrc.processors.preprocessors.natural_questions import NaturalQuestionsPreProcessor
from primeqa.mrc.processors.postprocessors.natural_questions import NaturalQuestionsPostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from tests.primeqa.mrc.common.base import UnitTest


class TestNQPostProcessor(UnitTest):
    
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

    @pytest.fixture(scope='session')
    def mrc_output(self):
        start_scores = torch.FloatTensor([[-0.5, 0.0, 0.1, -0.1, -0.2, 0.3, 0.0, 0.0, 0.3, 0.2, 0.1, 0.1, 0.7, 1.5, 0.2, 0.1, 0.1, -1.0, 0, 0, 0]])
        end_scores = torch.FloatTensor([[-0.5, -0.3, -0.2, -0.5, -0.1, 0, 0.1, -0.5, 0.3, 0, 0, 0, 1.0, 2.0, 0.3, 0.5, 0.1, -1.0, 0, 0, 0]])
        type_scores = torch.FloatTensor([[0.2, 0.2, 0.2, 0.2, 0.2]])
        return (start_scores, end_scores, type_scores)

    @pytest.fixture(scope='session')
    def expected_span_answer_after_adjustment(self):
        answer_text = 'Bob'
        start_position = 37
        end_position = 40
        return answer_text, start_position, end_position

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

    @pytest.fixture(scope='class')
    def nq_eval_examples_and_features(self, nq_examples, nq_preprocessor):
        return nq_preprocessor.process_eval(nq_examples)


    def test_post_processor_has_examples_and_features(self, nq_eval_examples_and_features,
                                                      mrc_output,
                                                      expected_span_answer_after_adjustment):
        scorer_type='weighted_sum_target_type_and_score_diff'
        postprocessor_class = NaturalQuestionsPostProcessor(k=1, n_best_size=1, max_answer_length=20,
                                                            scorer_type=SupportedSpanScorers(scorer_type),
                                                            single_context_multiple_passages=True)
        eval_examples, eval_features = nq_eval_examples_and_features
        nq_predictions = postprocessor_class.process(eval_examples, eval_features, mrc_output)
        span_answer_text, span_start_position, span_end_position = expected_span_answer_after_adjustment

        assert nq_predictions["000"][0]["span_answer"]["start_position"] == span_start_position
        assert nq_predictions["000"][0]["span_answer"]["end_position"] == span_end_position
        assert nq_predictions["000"][0]["span_answer_text"] == span_answer_text
