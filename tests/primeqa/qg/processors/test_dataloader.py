import pytest
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader


@pytest.mark.parametrize("model_name",["t5-small"])
def test_qg_dataloader(model_name):
    # model might change tokenizer
    tokenizer = QGModel(model_name, modality='table').tokenizer
    dataset_name="wikisql"
    modality = "table"
    max_len=200
    target_max_len=40

    qgdl = QGDataLoader(
        tokenizer=tokenizer,
        modality=modality,
        dataset_name=dataset_name,
        input_max_len=max_len,
        target_max_len=target_max_len
    )

    valid_dataset = qgdl.create(data_split="validation[:50]")
    
    assert (len(valid_dataset)>0)

    

