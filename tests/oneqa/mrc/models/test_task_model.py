import pytest
import torch
from pytest import raises
from transformers import AutoConfig, MODEL_FOR_PRETRAINING_MAPPING

from oneqa.mrc.models.task_model import ModelForDownstreamTasks
from oneqa.mrc.models.heads.extractive import ExtractiveQAHead, EXTRACTIVE_HEAD
from oneqa.mrc.types.model_outputs.extractive import ExtractiveQAModelOutput
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.common.base import UnitTest
from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_FIXTURE_WITH_MODEL_NAME


class TestModelForDownstreamTasks(UnitTest):
    @PARAMETERIZE_FIXTURE_WITH_MODEL_NAME
    def config_and_model_with_extractive_head(self, request):
        model_name = request.param
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        head_name = next(iter(EXTRACTIVE_HEAD))
        model.set_task_head(head_name)
        return config, model

    @pytest.fixture(scope='class')
    def training_inputs(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        kwargs = dict(start_positions=torch.randint(0, seq_len, (bs, 1)),
                      end_positions=torch.randint(0, seq_len, (bs, 1)),
                      target_type=torch.randint(0, len(TargetType), (bs, 1)))
        kwargs.update(model.dummy_inputs)
        return kwargs

    def test_from_config(self, config_and_model_with_extractive_head):
        config, model = config_and_model_with_extractive_head
        head_name = next(iter(EXTRACTIVE_HEAD))
        assert isinstance(model.task_heads[head_name], ExtractiveQAHead)
        assert isinstance(model, ModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    def test_model_class_from_config_then_from_pretrained(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        model_class = ModelForDownstreamTasks.model_class_from_config(config)
        model = model_class.from_pretrained(model_name,
                                            config=config,
                                            task_heads=EXTRACTIVE_HEAD)
        head_name = next(iter(EXTRACTIVE_HEAD))
        assert isinstance(model.task_heads[head_name], ExtractiveQAHead)
        assert isinstance(model, ModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    def test_raises_type_error_on_direct_instantiation(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        with raises(TypeError):
            _ = ModelForDownstreamTasks.from_pretrained(model_name,
                                                        config=config,
                                                        task_heads=EXTRACTIVE_HEAD)

    def test_raises_value_error_on_empty_task_head_dict(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        with raises(ValueError):
            _ = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads={})

    def test_model__property(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        assert model.model_ is getattr(model, model.base_model_prefix)

    def test_task_head_property_raises_value_error_when_not_set(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        with raises(ValueError):
            _ = model.task_head

    def test_task_head_property_retreives_set_task_head(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        head_name = next(iter(EXTRACTIVE_HEAD))
        model.set_task_head(head_name)
        assert model.task_head is model.task_heads[head_name]

    def test_set_task_head_raises_key_error_on_invalid_name(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        head_name = 'NOT_A_REAL_NAME'
        with raises(KeyError):
            model.set_task_head(head_name)

    def test_set_head_multiple_times(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        task_heads = dict(qa_head=ExtractiveQAHead, qa_head_2=ExtractiveQAHead)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=task_heads)
        for _ in range(5):
            for head_name in task_heads:
                model.set_task_head(head_name)
                assert model.task_head is model.task_heads[head_name]

    def test_forward_with_extractive_head(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        results = model(**model.dummy_inputs)

        assert isinstance(results, ExtractiveQAModelOutput)

        assert results.loss is None

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

    def test_forward_for_training_with_extractive_head(self, config_and_model_with_extractive_head, training_inputs):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        results = model(**training_inputs)

        assert isinstance(results, ExtractiveQAModelOutput)

        assert isinstance(results.loss.item(), float)

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

    def test_forward_with_extractive_head_with_tuple_output(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        results = model(**model.dummy_inputs, return_dict=False)

        assert isinstance(results, tuple)

        start_logits = results[0]
        self._assert_is_floating_point_tensor(start_logits)
        assert start_logits.shape == (bs, seq_len)

        end_logits = results[1]
        self._assert_is_floating_point_tensor(end_logits)
        assert end_logits.shape == (bs, seq_len)

        target_type_logits = results[2]
        self._assert_is_floating_point_tensor(target_type_logits)
        assert target_type_logits.shape == (bs, len(TargetType))

    def test_forward_for_training_with_extractive_head_with_tuple_output(
            self, config_and_model_with_extractive_head, training_inputs):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        results = model(**training_inputs, return_dict=False)

        assert isinstance(results, tuple)

        loss = results[0]
        assert isinstance(loss.item(), float)

        start_logits = results[1]
        self._assert_is_floating_point_tensor(start_logits)
        assert start_logits.shape == (bs, seq_len)

        end_logits = results[2]
        self._assert_is_floating_point_tensor(end_logits)
        assert end_logits.shape == (bs, seq_len)

        target_type_logits = results[3]
        self._assert_is_floating_point_tensor(target_type_logits)
        assert target_type_logits.shape == (bs, len(TargetType))
