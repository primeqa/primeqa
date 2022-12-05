import pytest
from primeqa.components.reader_component import ReaderComponent
from primeqa.tableqa.tapex.models.tapex_model import TapexModel
from transformers import (
    BartForConditionalGeneration,
    TapexTokenizer
)

@pytest.mark.parametrize("config_path",["../../primeqa/tableqa/tapex/tapex_config.json"])
def test_tapex_model(config_path):
    tapex_model = TapexModel(config_path)
    tapex_model.load_model_from_config(config_path)
    reader = ReaderComponent('TapexModel',"../../primeqa/tableqa/tapex/tapex_config.json")
    assert reader._class_object=="TapexModel"
    assert type(tapex_model.model)== BartForConditionalGeneration
    assert type(tapex_model.tokenizer) == TapexTokenizer
    
    
