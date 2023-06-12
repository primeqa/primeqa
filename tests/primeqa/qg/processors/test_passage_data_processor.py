import pytest
from datasets import load_dataset, Dataset
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.passage_qg.qg_processor import QGProcessor


@pytest.mark.parametrize("model_name",["t5-small"])
def test_preprocess_data_for_qg(model_name):
    # model might change tokenizer
    tokenizer = QGModel(model_name, modality='passage').tokenizer
    input_max_len = 1024
    target_max_len = 1024

    processor = QGProcessor(tokenizer, input_max_len, target_max_len)
    dataset = load_dataset('squad', split='validation')
    processed_dataset = processor(dataset)
    assert processed_dataset != None
    assert len(processed_dataset['question']) == len(processed_dataset['input'])

    processor = QGProcessor(tokenizer, input_max_len, target_max_len)
    raw_dataset = load_dataset('tydiqa', name='secondary_task', split='validation', streaming=True)
    iterable_dataset = raw_dataset.take(100)
    examples = {}
    for e in iterable_dataset:
        for key in e.keys():
            if key not in examples:
                examples[key] = []
            examples[key].append(e[key])
    dataset =  Dataset.from_dict(examples)
    
    processed_dataset = processor(dataset)
    assert processed_dataset != None
    assert len(processed_dataset['question']) == len(processed_dataset['input'])
