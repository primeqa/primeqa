from transformers import TapasConfig,TapasTokenizer, TapasForQuestionAnswering
import pandas as pd
import pytest
from primeqa.tableqa.tapas.tapas_component import TapasReader
import unittest

from primeqa.tableqa.tapas.metrics.answer_accuracy import compute_denotation_accuracy
from primeqa.tableqa.tapas.trainers.tableqa_trainer import TableQATrainer
from transformers import TapasConfig
from transformers import (
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,default_data_collator,
)
from primeqa.mrc.run_mrc import ModelArguments, DataTrainingArguments
from primeqa.tableqa.tapas.utils.tapas_args import TableQAArguments
from primeqa.tableqa.tapas.utils.data_collator import TapasCollator
from primeqa.tableqa.tapas.preprocessors.wikisql_preprocessor import load_data
from primeqa.tableqa.tapas.postprocessor.wikisql import WikiSQLPostprocessor
from primeqa.tableqa.tapas.metrics.answer_accuracy import compute_denotation_accuracy
import os

def test_tableqa_model():
    config_json_path= "./primeqa/tableqa/tapas/configs/tapas_config.json"
    reader = TapasReader(config_json_path)
    assert type(reader.model)==TapasForQuestionAnswering
    assert type(reader.tokenizer)==TapasTokenizer
    model = reader.model
    tokenizer = reader.tokenizer
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments,TableQAArguments))
    model_args, data_args, training_args, tqa_args = parser.parse_json_file(json_file=os.path.abspath(config_json_path)) 


    

    dataset_name="wikisql" 
    data_path_root="data/wikisql/" 
    output_dir="../../models/tableqa/wikisql_nb"

    tqa_args = TableQAArguments()
    tqa_args.data_path_root=data_path_root
    
    # config = TapasConfig(tqa_args)
    # tableqa_model = TableQAModel("google/tapas-base",config=config)
    # assert type(tableqa_model.model)==TapasForQuestionAnswering
    # assert type(tableqa_model.tokenizer)==TapasTokenizer
    # model = tableqa_model.model
    # tokenizer = tableqa_model.tokenizer

    post_obj = WikiSQLPostprocessor(tokenizer,tqa_args)

    train_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
    )

    train_dataset,eval_dataset = load_data(tqa_args.data_path_root,tokenizer,10,10)
    trainer = TableQATrainer(model=model,
                                args=train_args,
                                train_dataset=train_dataset if train_args.do_train else None,
                                eval_dataset=eval_dataset if train_args.do_eval else None,
                                tokenizer=reader.tokenizer,
                                data_collator=TapasCollator(),
                                post_process_function= post_obj.postprocess_prediction,
                                compute_metrics=compute_denotation_accuracy  
                                )
    train_result = trainer.train()
    assert(train_result!=None)
    metrics = train_result.metrics
    assert(metrics['epoch']>0)
    assert(metrics['train_loss']<10.0)

    eval_metrics= trainer.evaluate()
    assert(eval_metrics['epoch']>0)
    

