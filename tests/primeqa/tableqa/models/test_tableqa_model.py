from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import pytest
from primeqa.tableqa.models.tableqa_model import TableQAModel

from primeqa.tableqa.metrics.answer_accuracy import compute_denotation_accuracy
from primeqa.tableqa.models.tableqa_model import TableQAModel
from primeqa.tableqa.trainer.tableqa_trainer import TableQATrainer
from transformers import TapasConfig
from transformers import (
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,default_data_collator,
)

from primeqa.tableqa.run_tableqa import TableQAArguments
from primeqa.tableqa.utils.data_collator import TapasCollator
from primeqa.tableqa.preprocessors.wikisql_preprocessor import load_data

@pytest.mark.parametrize("model_name_path",["google/tapas-base"])
def test_tableqa_model(model_name_path):
    config=None
    tqam = TableQAModel("google/tapas-base",config=config)
    assert type(tqam.model)==TapasForQuestionAnswering
    assert type(tqam.tokenizer)==TapasTokenizer

    dataset_name="wikisql" 
    data_path_root="data/wikisql/" 
    output_dir="../../models/tableqa/wikisql_nb"

    tqa_args = TableQAArguments()
    tqa_args.dataset_name=dataset_name
    tqa_args.data_path_root=data_path_root
    config = TapasConfig(tqa_args)
    tableqa_model = TableQAModel("google/tapas-base",config=config)
    assert type(tableqa_model.model)==TapasForQuestionAnswering
    assert type(tableqa_model.tokenizer)==TapasTokenizer
    model = tableqa_model.model
    tokenizer = tableqa_model.tokenizer

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
    )

    train_dataset,eval_dataset = load_data(tqa_args.data_path_root,tokenizer,10,5)
    trainer = TableQATrainer(model=model,
                                args=train_args,
                                train_dataset=train_dataset if train_args.do_train else None,
                                eval_dataset=eval_dataset if train_args.do_eval else None,
                                tokenizer=tableqa_model.tokenizer,
                                data_collator=TapasCollator(),
                                )
    train_result = trainer.train()
    assert(train_result!=None)
