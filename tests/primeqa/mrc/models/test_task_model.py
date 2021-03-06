from pytest import raises
from transformers import MODEL_FOR_PRETRAINING_MAPPING

from primeqa.mrc.models.heads.extractive import ExtractiveQAHead, EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.data_models.model_outputs.extractive import ExtractiveQAModelOutput
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest


class TestModelForDownstreamTasks(UnitTest):
    def test_from_config(self, config_and_model_with_extractive_head):
        config, model = config_and_model_with_extractive_head
        head_name = next(iter(EXTRACTIVE_HEAD))
        assert isinstance(model.task_heads[head_name], ExtractiveQAHead)
        assert isinstance(model, ModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    def test_model_class_from_config_then_from_pretrained(self, model_name_and_config):
        model_name, config = model_name_and_config
        model_class = ModelForDownstreamTasks.model_class_from_config(config)
        model = model_class.from_pretrained(model_name,
                                            config=config,
                                            task_heads=EXTRACTIVE_HEAD)
        head_name = next(iter(EXTRACTIVE_HEAD))
        assert isinstance(model.task_heads[head_name], ExtractiveQAHead)
        assert isinstance(model, ModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    def test_raises_type_error_on_direct_instantiation(self, model_name_and_config):
        model_name, config = model_name_and_config
        with raises(TypeError):
            _ = ModelForDownstreamTasks.from_pretrained(model_name,
                                                        config=config,
                                                        task_heads=EXTRACTIVE_HEAD)

    def test_raises_value_error_on_empty_task_head_dict(self, model_name_and_config):
        model_name, config = model_name_and_config
        with raises(ValueError):
            _ = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads={})

    def test_model__property(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        assert model.model_ is getattr(model, model.base_model_prefix)

    def test_task_head_property_raises_value_error_when_not_set(self, model_name_and_config):
        model_name, config = model_name_and_config
        model = ModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=EXTRACTIVE_HEAD)
        with raises(ValueError):
            _ = model.task_head

    def test_task_head_property_retreives_set_task_head(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        head_name = next(iter(EXTRACTIVE_HEAD))
        model.set_task_head(head_name)
        assert model.task_head is model.task_heads[head_name]

    def test_set_task_head_raises_key_error_on_invalid_name(self, config_and_model_with_extractive_head):
        _, model = config_and_model_with_extractive_head
        head_name = 'NOT_A_REAL_NAME'
        with raises(KeyError):
            model.set_task_head(head_name)

    def test_set_head_multiple_times(self, model_name_and_config):
        model_name, config = model_name_and_config
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

    def test_forward_for_training_with_extractive_head(self, config_and_model_with_extractive_head,
                                                       extractive_training_inputs):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        results = model(**extractive_training_inputs)

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
            self, config_and_model_with_extractive_head, extractive_training_inputs):
        _, model = config_and_model_with_extractive_head
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        results = model(**extractive_training_inputs, return_dict=False)

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
