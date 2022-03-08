import pytest
from pytest import raises
from transformers import AutoConfig, MODEL_FOR_PRETRAINING_MAPPING

from oneqa.mrc.models.task_model import ModelForDownstreamTasks
from oneqa.mrc.models.heads.extractive import ExtractiveQAHead, EXTRACTIVE_HEAD
from tests.oneqa.mrc.unit_test import UnitTest


class TestModelForDownstreamTasks(UnitTest):
    @UnitTest.PARAMETERIZE_MODEL_NAME
    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    def test_from_config(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        head_name = next(iter(EXTRACTIVE_HEAD))
        assert isinstance(model.task_heads[head_name], ExtractiveQAHead)
        assert isinstance(model, ModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    @pytest.mark.flaky(reruns=5, reruns_delay=2)
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

    @UnitTest.PARAMETERIZE_MODEL_NAME
    def test_model__property(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
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