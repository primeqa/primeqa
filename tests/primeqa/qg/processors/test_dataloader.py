import pytest
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer, AutoModelForSeq2SeqLM

@pytest.mark.parametrize("model_name",["t5-small"])
def test_qg_dataloader(model_name):
    dataset_name="wikisql"
    max_len=200
    target_max_len=40
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    qgdl = QGDataLoader(
        tokenizer=tokenizer,
        dataset_name=dataset_name,
        input_max_len=max_len,
        target_max_len=target_max_len
    )

    valid_dataset = qgdl.create("validation[:50]")
    
    assert (len(valid_dataset)>0)

    

