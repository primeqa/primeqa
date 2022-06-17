import pytest
import torch
from transformers import MODEL_MAPPING

from primeqa.mrc.models.heads.extractive import ExtractiveQAWithConfidenceHead
from primeqa.mrc.data_models.model_outputs.extractive import ExtractiveQAWithConfidenceModelOutput
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest


class TestExtractiveQAWithConfidenceHead(UnitTest):
    @pytest.fixture(scope='session')
    def config_and_language_model(self, model_name_and_config):
        model_name, config = model_name_and_config
        model = MODEL_MAPPING[config.__class__].from_pretrained(model_name, config=config)
        return config, model

    @pytest.fixture(scope='session')
    def extractive_with_confidence_head(self, model_name_and_config):
        _, config = model_name_and_config
        return ExtractiveQAWithConfidenceHead(config)

    @pytest.fixture(scope='session')
    def language_model_outputs(self, config_and_language_model):
        _, model = config_and_language_model
        return model(**model.dummy_inputs)

    @pytest.fixture(scope='session')
    def language_model_outputs_tuple(self, language_model_outputs):
        return language_model_outputs.to_tuple()

    @pytest.fixture(scope='session')
    def training_inputs(self, config_and_language_model, language_model_outputs):
        _, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        kwargs = dict(start_positions=torch.randint(0, seq_len, (bs, 1)),
                      end_positions=torch.randint(0, seq_len, (bs, 1)),
                      target_type=torch.randint(0, len(TargetType), (bs, 1)))
        args = (language_model_outputs,)
        return args, kwargs

    @pytest.fixture(scope='session')
    def eval_inputs_with_attention_and_token_type(self, config_and_language_model, language_model_outputs):
        _, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        position_ids = torch.arange(seq_len).repeat(bs, 1)
        context_start_positions = torch.randint(1, seq_len, (bs, 1))
        token_type_ids = (position_ids >= context_start_positions).long()
        attention_mask = torch.ones(bs, seq_len, dtype=torch.long)
        kwargs = dict(attention_mask=attention_mask,
                      token_type_ids=token_type_ids)
        args = (language_model_outputs,)
        return args, kwargs

    @pytest.fixture(scope='session')
    def eval_inputs_with_input_ids(self, config_and_language_model, language_model_outputs):
        _, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape        
        kwargs = dict(input_ids=model.dummy_inputs['input_ids'])
        args = (language_model_outputs,)
        return args, kwargs
    
    @pytest.fixture(scope='session')
    def training_inputs_with_tuple_from_language_model(self, training_inputs):
        args, kwargs = training_inputs
        args = (args[0].to_tuple(),)
        return args, kwargs

    def test_instantiation(self, model_name_and_config):
        _, config = model_name_and_config
        head = ExtractiveQAWithConfidenceHead(config)
        assert head.num_labels == config.num_labels

    def test_correct_number_of_classification_labels_when_using_default(self, extractive_with_confidence_head):
        assert extractive_with_confidence_head.num_classification_head_labels == len(TargetType)

    def test_correct_number_of_classification_labels_when_overridden(self, model_name_and_config):
        _, config = model_name_and_config
        num_classification_labels = 16
        head = ExtractiveQAWithConfidenceHead(config, num_labels_override=num_classification_labels)
        assert head.num_classification_head_labels == num_classification_labels

    def test_config_when_overridden(self, model_name_and_config):
        _, config = model_name_and_config
        config.output_dropout_rate = 0.3
        config.decoding_times_with_dropout = 10
        config.sep_token_id = 0
        head = ExtractiveQAWithConfidenceHead(config)
        assert head.output_dropout_rate == 0.3
        assert head.decoding_times_with_dropout == 10
        assert head.sep_token_id == 0

    def test_forward(self, config_and_language_model, language_model_outputs):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAWithConfidenceHead(config)
        results = head(language_model_outputs)

        assert isinstance(results, ExtractiveQAWithConfidenceModelOutput)

        assert results.loss is None

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

        self._assert_is_floating_point_tensor(results.start_stdev)
        assert results.start_stdev.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_stdev)
        assert results.end_stdev.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.query_passage_similarity)
        assert torch.equal(results.query_passage_similarity, torch.zeros(bs, dtype=torch.float))

        
    def test_forward_for_training(self, config_and_language_model, training_inputs):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAWithConfidenceHead(config)

        args, kwargs = training_inputs
        results = head(*args, **kwargs)

        assert isinstance(results, ExtractiveQAWithConfidenceModelOutput)

        assert isinstance(results.loss.item(), float)

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

        assert results.start_stdev is None
        assert results.end_stdev is None
        assert results.query_passage_similarity is None
        
    def test_forward_for_eval_with_attention_and_token_type(self, config_and_language_model, eval_inputs_with_attention_and_token_type):
        config, model = config_and_language_model
        config.output_dropout_rate = 0.3
        config.decoding_times_with_dropout = 10
        config.sep_token_id = 0
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAWithConfidenceHead(config)

        args, kwargs = eval_inputs_with_attention_and_token_type
        results = head(*args, **kwargs)

        assert isinstance(results, ExtractiveQAWithConfidenceModelOutput)

        assert results.loss is None

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

        self._assert_is_floating_point_tensor(results.start_stdev)
        assert results.start_stdev.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_stdev)
        assert results.end_stdev.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.query_passage_similarity)
        assert results.query_passage_similarity.shape == torch.Size([bs])


    def test_forward_for_eval_with_input_ids(self, config_and_language_model, eval_inputs_with_input_ids):
        config, model = config_and_language_model
        config.output_dropout_rate = 0.3
        config.decoding_times_with_dropout = 10
        config.sep_token_id = 0
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAWithConfidenceHead(config)

        args, kwargs = eval_inputs_with_input_ids
        results = head(*args, **kwargs)

        assert isinstance(results, ExtractiveQAWithConfidenceModelOutput)

        assert results.loss is None

        self._assert_is_floating_point_tensor(results.start_logits)
        assert results.start_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_logits)
        assert results.end_logits.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.target_type_logits)
        assert results.target_type_logits.shape == (bs, len(TargetType))

        self._assert_is_floating_point_tensor(results.start_stdev)
        assert results.start_stdev.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.end_stdev)
        assert results.end_stdev.shape == (bs, seq_len)

        self._assert_is_floating_point_tensor(results.query_passage_similarity)
        assert results.query_passage_similarity.shape == torch.Size([bs])


    def test_forward_with_tuple_input(self, config_and_language_model, language_model_outputs_tuple):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAWithConfidenceHead(config)
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

        start_stdev = results[3]
        self._assert_is_floating_point_tensor(start_stdev)
        assert start_stdev.shape == (bs, seq_len)

        end_stdev = results[4]
        self._assert_is_floating_point_tensor(end_stdev)
        assert end_stdev.shape == (bs, seq_len)

        query_passage_similarity = results[5]
        self._assert_is_floating_point_tensor(query_passage_similarity)
        assert torch.equal(query_passage_similarity, torch.zeros(bs, dtype=torch.float))

        
    def test_forward_for_training_with_tuple_input(self, config_and_language_model,
                                                   training_inputs_with_tuple_from_language_model):
        config, model = config_and_language_model
        bs, seq_len = model.dummy_inputs['input_ids'].shape
        head = ExtractiveQAWithConfidenceHead(config)

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

        assert len(results) == 4
