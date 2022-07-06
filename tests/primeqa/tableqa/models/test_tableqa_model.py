from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import pytest
from primeqa.tableqa.models.tableqa_model import TableQAModel

@pytest.mark.parametrize("model_name_path",["google/tapas-base"])
def test_tableqa_model(model_name_path):
    config=None
    tqam = TableQAModel("google/tapas-base",config=config)
    assert type(tqam.model)==TapasForQuestionAnswering
    assert type(tqam.tokenizer)==TapasTokenizer
