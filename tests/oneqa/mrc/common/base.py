import pytest
import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from oneqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from oneqa.mrc.models.task_model import ModelForDownstreamTasks
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

    @staticmethod
    def _assert_is_floating_point_tensor(tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype.is_floating_point
