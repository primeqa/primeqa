import pytest
from datasets import load_dataset
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.hybrid_qg.hybridqa_processor import HybridQAProcessor

@pytest.mark.parametrize("model_name",["t5-small"])
def test_preprocess_data_for_qg(model_name):
    # model might change tokenizer
    tokenizer = QGModel(model_name, modality='hybrid').tokenizer
    input_max_len = 512
    target_max_len = 30

    processor = HybridQAProcessor(tokenizer, input_max_len, target_max_len)
    dataset = load_dataset('hybrid_qa', split='validation[:100]')
    processed_dataset = processor(dataset)
    assert processed_dataset != None
    assert len(processed_dataset['label']) == len(processed_dataset['input'])
