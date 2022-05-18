import pytest
from oneqa.tableqg.models.tableqg_model import TableQG
from transformers import T5Tokenizer, T5ForConditionalGeneration


@pytest.mark.parametrize("model_name",["t5-base"])
def test_table_qg_model(model_name):
    tqm = TableQG(model_name)
    assert type(tqm.model)==T5ForConditionalGeneration
    assert type((tqm.tokenizer)==T5Tokenizer)
