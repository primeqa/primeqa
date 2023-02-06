import pytest
from primeqa.tableqa.tapex.tapex_component import TapexReader
from transformers import (
    BartForConditionalGeneration,
    TapexTokenizer
)

@pytest.mark.parametrize("config_path",["tests/resources/tapex/tapex_config.json"])
def test_tapex_model(config_path):
    reader = TapexReader(config_path)
    reader.load(config_path)
    assert type(reader.model)== BartForConditionalGeneration
    assert type(reader.tokenizer) == TapexTokenizer
    
    
    
    
