import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from primeqa.mrc.processors.preprocessors.tydiqa_google import TyDiQAGooglePreprocessor
from tests.primeqa.mrc.common.base import UnitTest


class TestTyDiQAGooglePreprocessor(UnitTest):

    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def eval_examples(self):
        # in the future test on file on disk
        # eval_files = "data/nq/nq_dev_tydiformat-00.jsonl"
        # examples = datasets.load_dataset('json', data_files={'dev':eval_files})['dev']
        question = ["how to add friends on snapchat near me?"]
        context = ["How do you get local singles on Snapchat?", 
        "To use Add Nearby, follow these steps: \n* Tap 'Add Friends'.\n* Tap 'Add Nearby' and select 'Ok' to allow Snapchat to use your location for finding nearby Snapchatters.\n* Ask a friend to open Add Nearby on their phone.\n*Tap the '+' sign next to your friend's username to add them! If a friend adds you, we'll let you know!\n",
        "Beside this, how do you find random people on Snapchat?", 
        "If you have other social media platform such as Facebook, Instagram and Twitter, you can use that to add random people there. Just use the search bar in the social media platform and type '#snapcode'. \nIt will then show you people who promoted their snapcode so that people can add them in their snapchat."]
        example_id = ["gooaq-ccqa-ex"]
        annotation = {}
        annotation['passage_answer'] = {'candidate_index':1}
        annotation['minimal_answer'] = {'plaintext_start_byte':82,'plaintext_end_byte':363}
        annotation['yes_no_answer'] = "NONE"
        passage_candidates = []
        start = 0
        end = 0
        for passage in context:
            end += len(passage.encode())
            passage_candidates.append({'plaintext_start_byte':start,'plaintext_end_byte':end})
            start+=end+1
        examples_dict = dict(question_text=question, document_plaintext=[" ".join(context)], example_id=example_id,
                                annotations=[[annotation]],passage_answer_candidates=[passage_candidates])
        examples = datasets.Dataset.from_dict(examples_dict)
        return examples

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def tydiqa_google_preprocessor(self, tokenizer):
        return TyDiQAGooglePreprocessor(
            tokenizer,
            stride=256,
            load_from_cache_file=False,
        )

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, tydiqa_google_preprocessor):
        eval_examples, eval_features = tydiqa_google_preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        for example in eval_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
