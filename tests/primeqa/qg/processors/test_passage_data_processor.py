import pytest
from datasets import load_dataset
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.passage_qg.qa2s_processor import QA2SProcessor
from primeqa.qg.processors.passage_qg.qg_processor import QGProcessor


@pytest.mark.parametrize("model_name",["t5-small"])
def test_preprocess_data_for_qg(model_name):
    # model might change tokenizer
    tokenizer = QGModel(model_name, modality='passage_qg').tokenizer
    input_max_len = 1024
    target_max_len = 1024

    processor = QGProcessor(tokenizer, input_max_len, target_max_len)
    dataset = load_dataset('squad', split='validation')
    processed_dataset = processor(dataset)
    assert processed_dataset != None
    assert len(processed_dataset['question']) == len(processed_dataset['input'])

    processor = QGProcessor(tokenizer, input_max_len, target_max_len)
    dataset = load_dataset('tydiqa', name='secondary_task', split='validation')
    processed_dataset = processor(dataset)
    assert processed_dataset != None
    assert len(processed_dataset['question']) == len(processed_dataset['input'])

@pytest.mark.parametrize("model_name",["t5-small"])
def test_preprocess_data_for_qa2s(model_name):
    # model might change tokenizer
    tokenizer = QGModel(model_name, modality='passage_qa2s').tokenizer
    processor = QA2SProcessor(tokenizer)
    dataset = load_dataset('squad', split='validation')
    processed_dataset = processor(dataset)
    assert processed_dataset!=None
    assert len(processed_dataset['question']) == len(processed_dataset['input_ids'])
