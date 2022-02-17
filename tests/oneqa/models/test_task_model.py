import pytest
from pytest import raises
from transformers import AutoConfig

from oneqa.mrc.models.task_model import create_task_model_class_from_config, ModelForDownstreamTasks
from oneqa.mrc.models.heads.extractive import ExtractiveQAHead


class TestModelForDownstreamTasks:

    @pytest.mark.parametrize("model_name",
                             ['roberta-base',
                              'xlm-roberta-base',
                              'bert-base-uncased',
                              'albert-base-v2',
                              'facebook/bart-base'])
    def test_from_pretrained(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        model_class = create_task_model_class_from_config(config)
        model = model_class.from_pretrained(model_name,
                                            config=config,
                                            task_heads=dict(qa_head=ExtractiveQAHead))
        assert isinstance(model.task_heads['qa_head'], ExtractiveQAHead)

    def test_raises_type_error_on_direct_instantiation(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        with raises(TypeError):
            _ = ModelForDownstreamTasks.from_pretrained(model_name,
                                                        config=config,
                                                        task_heads={})
