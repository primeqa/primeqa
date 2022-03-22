import pytest
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from oneqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from oneqa.mrc.models.task_model import ModelForDownstreamTasks
from oneqa.mrc.processors.preprocessors.default import DefaultPreProcessor
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_FIXTURE_WITH_MODEL_NAME


class UnitTest:
    """
    Base class for all unit test classes
    """

    @PARAMETERIZE_FIXTURE_WITH_MODEL_NAME
    def model_name_and_config(self, request):
        model_name = request.param
        _ = AutoModel.from_pretrained(model_name)  # Pre-download LM inside flaky fixture so other tests have it
        return model_name, AutoConfig.from_pretrained(model_name)

    @pytest.mark.flaky(reruns=5, reruns_delay=2)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def tokenizer(self, model_name_and_config):
        model_name, _ = model_name_and_config
        return AutoTokenizer.from_pretrained(model_name)

    @pytest.fixture(scope='session')
    def config_and_model_with_extractive_head(self, model_name_and_config):
        model_name, config = model_name_and_config
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        head_name = next(iter(EXTRACTIVE_HEAD))
        model.set_task_head(head_name)
        return config, model

    @pytest.fixture(scope='session')
    def extractive_training_inputs(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        kwargs = dict(start_positions=torch.randint(0, seq_len, (bs, 1)),
                      end_positions=torch.randint(0, seq_len, (bs, 1)),
                      target_type=torch.randint(0, len(TargetType), (bs, 1)))
        kwargs.update(model.dummy_inputs)
        return kwargs

    @pytest.fixture(scope='session')
    def train_examples(self):
        question = ["Who walked the dog?", "What time is it?"]
        context = [["Alice walks the cat", "Bob walks the dog"],
                   ["The quick brown fox jumps over the lazy dog", "Glenn the otter lives at the aquarium", "Go"]]
        example_id = ["foo-abc", "bar-123"]
        start_positions = [[0], [-1]]
        end_positions = [[2], [-1]]
        passage_indices = [[1], [-1]]
        yes_no_answer = [["NONE"], ["NONE"]]
        examples_dict = dict(question=question, context=context, example_id=example_id,
                             target=[dict(start_positions=s, end_positions=e, passage_indices=p, yes_no_answer=yn)
                                     for s, e, p, yn in
                                     zip(start_positions, end_positions, passage_indices, yes_no_answer)])
        examples_dataset = Dataset.from_dict(examples_dict)
        return examples_dataset

    @pytest.fixture(scope='session')
    def eval_examples(self, train_examples):
        return train_examples.remove_columns('target')

    @pytest.fixture(scope='session')
    def preprocessor(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=0.5,
            negative_sampling_prob_when_no_answer=0.5,
        )

    @pytest.fixture(scope='session')
    def train_examples_and_features(self, train_examples, preprocessor):
        return preprocessor.process_train(train_examples)

    @pytest.fixture(scope='session')
    def eval_examples_and_features(self, eval_examples, preprocessor):
        return preprocessor.process_eval(eval_examples)

    @staticmethod
    def _assert_is_floating_point_tensor(tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype.is_floating_point
