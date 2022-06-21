import pytest
from oneqa.qg.models.qg_model import QGModel
from transformers import T5Tokenizer, T5ForConditionalGeneration


@pytest.mark.parametrize("model_name",["t5-base"])
def test_qg_model(model_name):
    tqm = QGModel(model_name, modality='table')
    assert type(tqm.model)==T5ForConditionalGeneration
    assert type((tqm.tokenizer)==T5Tokenizer)
