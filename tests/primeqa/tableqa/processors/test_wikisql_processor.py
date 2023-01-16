from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import pytest
from primeqa.tableqa.tapas.models.tapas_model import TapasModel
from primeqa.tableqa.tapas.tapas_component import TapasReader
from primeqa.tableqa.tapas.preprocessors.wikisql_preprocessor import load_data

@pytest.mark.parametrize("output_dir",["."])
def test_tableqa_model(output_dir):
    config_json_path= "./primeqa/tableqa/tapas/configs/tapas_config.json"
    reader = TapasReader(config_json_path)
    tokenizer = reader.tokenizer
    train_dataset={}
    eval_dataset={}
    train_dataset,eval_dataset = load_data(output_dir,tokenizer,10,5)
    assert(len(train_dataset)>0)
    assert(len(eval_dataset)>0)

