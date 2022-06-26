import pytest
from primeqa.qg.models.qg_model import QGModel
from transformers import T5Tokenizer, T5ForConditionalGeneration


@pytest.mark.parametrize("model_name",["t5-base"])
def test_qg_model(model_name):
    tqm = QGModel(model_name, modality='passage')
    assert type(tqm.model)==T5ForConditionalGeneration
    assert type((tqm.tokenizer)==T5Tokenizer)
