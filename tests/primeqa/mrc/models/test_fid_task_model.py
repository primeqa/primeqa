from pytest import raises
from transformers import MODEL_FOR_PRETRAINING_MAPPING
from transformers import AutoConfig, AutoModel, AutoTokenizer
from primeqa.mrc.models.heads.generative import FID_HEAD
from primeqa.mrc.models.fid_task_model import FiDModelForDownstreamTasks
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest
import pytest

class TestFIDModelForDownstreamTasks(UnitTest):
    @pytest.fixture(scope='session')
    def model_name_and_config(self):
        model_name = "facebook/bart-base"
        _ = AutoModel.from_pretrained(model_name)  # Pre-download LM inside flaky fixture so other tests have it
        return model_name, AutoConfig.from_pretrained(model_name)
    
    @pytest.fixture(scope='session')
    def config_and_model_with_generative_head(self, model_name_and_config):
        model_name, config = model_name_and_config
        model = FiDModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=FID_HEAD)
        head_name = next(iter(FID_HEAD))
        model.set_task_head(head_name)
        return config, model
    
    def test_from_config(self, config_and_model_with_generative_head):
        config, model = config_and_model_with_generative_head
        head_name = next(iter(FID_HEAD))
        assert isinstance(model, FiDModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    def test_model_class_from_config_then_from_pretrained(self, model_name_and_config):
        model_name, config = model_name_and_config
        model_class = FiDModelForDownstreamTasks.model_class_from_config(config)
        model = model_class.from_pretrained(model_name,
                                            config=config,
                                            task_heads=FID_HEAD)
        head_name = next(iter(FID_HEAD))
        assert isinstance(model, FiDModelForDownstreamTasks)
        assert isinstance(model, MODEL_FOR_PRETRAINING_MAPPING[config.__class__])

    def test_raises_type_error_on_direct_instantiation(self, model_name_and_config):
        model_name, config = model_name_and_config
        with raises(TypeError):
            _ = FiDModelForDownstreamTasks.from_pretrained(model_name,
                                                        config=config,
                                                        task_heads=FID_HEAD)


    def test_model__property(self, config_and_model_with_generative_head):
        _, model = config_and_model_with_generative_head
        assert model.model is getattr(model, model.base_model_prefix)


    def test_forward(self, config_and_model_with_generative_head):
        _, model = config_and_model_with_generative_head
        # make FiD input with one passage
        fid_inputs={}
        fid_inputs['input_ids'] = model.dummy_inputs['input_ids'][:,None,:]
        fid_inputs['attention_mask'] = model.dummy_inputs['attention_mask'][:,None,:]
        bs, passages, seq_len = fid_inputs['input_ids'].shape
        results = model(**fid_inputs)

        assert isinstance(results, tuple)
        
        self._assert_is_floating_point_tensor(results[0])
        assert results[0].shape == (bs, seq_len, model.config.vocab_size)