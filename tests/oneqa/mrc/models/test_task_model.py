import pytest
from pytest import raises
from transformers import AutoConfig, MODEL_FOR_PRETRAINING_MAPPING

from oneqa.mrc.models.task_model import ModelForDownstreamTasks
from oneqa.mrc.models.heads.extractive import ExtractiveQAHead, EXTRACTIVE_HEAD

_MODEL_NAMES = ("model_name",
                ['roberta-base', 'xlm-roberta-base', 'bert-base-uncased', 'albert-base-v2', 'facebook/bart-base'])


class TestModelForDownstreamTasks:
    @pytest.mark.parametrize(*_MODEL_NAMES)
    def test_from_config(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
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

    @pytest.mark.parametrize(*_MODEL_NAMES)
    def test_model_property(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        assert model.model_ is getattr(model, model.base_model_prefix)
