import pytest
import torch
from transformers import AutoConfig, MODEL_MAPPING

from oneqa.mrc.models.heads.extractive import ExtractiveQAHead
from oneqa.mrc.types.model_outputs.extractive import ExtractiveQAModelOutput
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.common.base import UnitTest
from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_TEST_WITH_MODEL_NAME, \
    PARAMETERIZE_FIXTURE_WITH_MODEL_NAME


class TestExtractiveQAHead(UnitTest):
    @PARAMETERIZE_FIXTURE_WITH_MODEL_NAME
    def config_and_language_model(self, request):
        model_name = request.param
        config = AutoConfig.from_pretrained(model_name)
        model = MODEL_MAPPING[config.__class__].from_pretrained(model_name, config=config)
        return config, model

    @pytest.fixture(scope='class')
    def language_model_outputs(self, config_and_language_model):
        _, model = config_and_language_model
        return model(**model.dummy_inputs)

    @pytest.fixture(scope='class')
    def language_model_outputs_tuple(self, language_model_outputs):
        return language_model_outputs.to_tuple()

    @pytest.fixture(scope='class')
    def training_inputs(self, config_and_language_model, language_model_outputs):
        _, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        kwargs = dict(start_positions=torch.randint(0, seq_len, (bs, 1)),
                      end_positions=torch.randint(0, seq_len, (bs, 1)),
                      target_type=torch.randint(0, len(TargetType), (bs, 1)))
        args = (language_model_outputs,)
        return args, kwargs

    @pytest.fixture(scope='class')
    def training_inputs_with_tuple_from_language_model(self, training_inputs):
        args, kwargs = training_inputs
        args = (args[0].to_tuple(),)
        return args, kwargs

    @PARAMETERIZE_TEST_WITH_MODEL_NAME
    def test_instantiation(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        head = ExtractiveQAHead(config)
        assert head.num_labels == config.num_labels

    def test_correct_number_of_classification_labels_when_using_default(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        head = ExtractiveQAHead(config)
        assert head.num_classification_head_labels == len(TargetType)

    def test_correct_number_of_classification_labels_when_overridden(self):
        model_name = 'roberta-base'
        num_classification_labels = 16
        config = AutoConfig.from_pretrained(model_name)
        head = ExtractiveQAHead(config, num_labels_override=num_classification_labels)
        assert head.num_classification_head_labels == num_classification_labels

    def test_forward(self, config_and_language_model, language_model_outputs):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAHead(config)
        results = head(language_model_outputs)

        assert isinstance(results, ExtractiveQAModelOutput)

        assert results.loss is None

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

    def test_forward_with_tuple_input(self, config_and_language_model, language_model_outputs_tuple):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAHead(config)
        results = head(language_model_outputs_tuple)

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

    def test_forward_for_training(self, config_and_language_model, training_inputs):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAHead(config)

        args, kwargs = training_inputs
        results = head(*args, **kwargs)

        assert isinstance(results, ExtractiveQAModelOutput)

        assert isinstance(results.loss.item(), float)

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

    def test_forward_for_training_with_tuple_input(self, config_and_language_model,
                                                   training_inputs_with_tuple_from_language_model):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAHead(config)

        args, kwargs = training_inputs_with_tuple_from_language_model
        results = head(*args, **kwargs)

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
