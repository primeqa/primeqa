from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import pytest
from primeqa.tableqa.models.tableqa_model import TableQAModel
from primeqa.tableqa.preprocessors.wikisql_preprocessor import load_data

@pytest.mark.parametrize("output_dir",["."])
def test_tableqa_model(output_dir):
    config=None
    tqam = TableQAModel("google/tapas-base",config=config)
    model = tqam.model
    tokenizer = tqam.tokenizer
    train_dataset={}
    eval_dataset={}
    train_dataset,eval_dataset = load_data(output_dir,tokenizer,10,5)
    assert(len(train_dataset)>0)
    assert(len(eval_dataset)>0)

