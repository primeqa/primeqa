from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from primeqa.mitqa.utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments
import json
from primeqa.mitqa.utils.partial_label_utils import vec_mat_multiplication,retrieval_accuracy
from primeqa.mitqa.utils.hybridqa_utils import tokenize
from primeqa.mitqa.utils.ottqa_utils import assign_ids
from primeqa.mitqa.utils.json_utils import read_data
from primeqa.mitqa.utils.model_utils.row_retriever_MITQA import RowRetriever
from primeqa.mitqa.utils.model_utils.reranker import re_rank_ae_output
from primeqa.mitqa.utils.model_utils.process_row_retriever_output import preprocess_data_using_row_retrieval_scores,preprocess_data_using_row_retrieval_scores_ottqa,create_dataset_for_answer_extractor
from primeqa.mitqa.utils.model_utils.answer_extractor_multi_Answer import predict_ae
from primeqa.mitqa.processors.preprocessors.preprocess_raw_data import preprocess_data,load_st_model
import logging
import torch
import os
import sys
import pytest
from primeqa.mitqa.utils.ottqa_utils  import assign_ids
from primeqa.mitqa.metrics.evaluate import normalize_answer,get_tokens,compute_exact,compute_f1
from primeqa.mitqa.metrics.evaluate import normalize_answer,get_tokens,compute_exact,compute_f1

import numpy as np
from primeqa.mitqa.mitqa_component import MITQAReader
from primeqa.mitqa.utils.create_table_retriever_training_data import linearize_row
from primeqa.mitqa.utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments
from primeqa.mitqa.utils.model_utils.table_retriever import train_table_retriever,predict_table_retriever
from primeqa.mitqa.utils.link_predictor import predict_link_for_tables,train_link_generator

hybridqa_config = {
    "per_device_train_batch_size_rr":8,
    "per_device_eval_batch_size_rr":8,
    "rr_model_name":"bert-base-uncased",
    "row_retriever_model_name_path":None,
    "pos_frac_per_epoch":[0.3, 0.3, 0.1, 0.0001, 0.0001],
    "group_frac_per_epoch":[0.0, 0.5, 1.0, 1.0, 1.0],
    "max_seq_length":512,
    "per_gpu_train_batch_size":8,
    "train_batch_size":8,
    "per_gpu_eval_batch_size":8,
    "eval_batch_size":8,
    "max_query_length":64,
    "threads":1,
    "null_score_diff_threshold":0.0,
    "n_best_size":20,
    "do_predict_ae":True,
    "n_gpu":1,
    "max_answer_length":30,
    "model_name_or_path_ae":"bert-base-uncased",
    "model_type":"bert",
    "doc_stride":128,
    "pred_ans_file":"tests/resources/mitqa/hybridqa/answer_extractor_output_test.json",
    "eval_file":"tests/resources/mitqa/hybridqa/ae_input_test.json",
    "output_dir":"tests/resources/mitqa/hybridqa/answer_extractor/",
    "model":"gpt2",
    "top_k":0,
    "top_p":0.9,
    "seed_lg":42,
    "batch_size_lg":2,
    "max_source_len":32,
    "max_target_len":16,
    "do_all_lg":True,
    "data_path_root":"tests/resources/mitqa/hybridqa/",
    "dataset_name":"hybridqa",
    "test_data_path":"tests/resources/mitqa/hybridqa/toy.json",
    "test":True
}
def test_hybirdqa():
    hqa_parser = HfArgumentParser((HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments))
    test=True
    hqa_args,lp_args,rr_args,ae_args,= hqa_parser.parse_dict(hybridqa_config)
    raw_test_data = json.load(open(hqa_args.test_data_path))
    doc_retriever = load_st_model()
    assert len(raw_test_data) !=0
    test_data_processed = preprocess_data(doc_retriever,hqa_args.data_path_root,hqa_args.dataset_name,raw_test_data,split="test",test=test)
    assert test_data_processed[0]['table_passage_row'] != None
    assert "table_row" in test_data_processed[0].keys()
    rr = RowRetriever(hqa_args,rr_args)
    qid_scores_dict = rr.predict(test_data_processed)
    assert qid_scores_dict != None
    test_processed_data = preprocess_data_using_row_retrieval_scores(doc_retriever,raw_test_data,qid_scores_dict,test)
    assert test_processed_data != None
    answer_extraction_data = create_dataset_for_answer_extractor(test_processed_data,hqa_args.data_path_root,test)
    assert answer_extraction_data != None
    ae_output_path,ae_output_path_nbest = predict_ae(ae_args,answer_extraction_data)
    assert ae_output_path != None and ae_output_path_nbest != None
    re_ranked_output = re_rank_ae_output(qid_scores_dict,ae_output_path_nbest,ae_args.pred_ans_file) 
    assert re_ranked_output!= None

lg_config = {
        "model":"gpt2",
        "learning_rate_lg":5e-5,
        "dataset":"tests/resources/mitqa/ottqa/train_dev_tables.json",
        "device_lg":torch.device("cpu"),
    }     
@pytest.mark.parametrize("lg_config",[lg_config])
def test_link_predictor(lg_config):
    hqa_parser = HfArgumentParser(LinkPredictorArguments)
    args= hqa_parser.parse_dict(lg_config)
    loss = train_link_generator(args[0])
    assert loss!=None
        
@pytest.mark.parametrize("test_string",["United %States %America"])
def test_hybridqa_utils_tokenize(test_string):
    tokenized = tokenize(test_string)
    assert tokenized=="United% States% America"

list1 = [0.5,0.3,1.1]
vec = torch.tensor(list1)
list2 = [[1, 4, 5, 12], 
    [-5, 8, 9, 0],
    [-6, 7, 11, 19]]
mat = torch.tensor(list2)
exp = torch.tensor([[ 0.5000,  2.0000,  2.5000,  6.0000],
        [-1.5000,  2.4000,  2.7000,  0.0000],
        [-6.6000,  7.7000, 12.1000, 20.9000]])
def test_partial_label_utils():
    res = vec_mat_multiplication(vec,mat)
    assert res!=None
    
data = [{"question_id":"abcd123"},{"question_id":"abcd123"},{"question_id":"abcd1234"}]
exp_res = [{"question_id":"abcd123_0"},{"question_id":"abcd123_1"},{"question_id":"abcd1234_0"}]
def test_ottqa_utils():
    res = assign_ids(data)
    assert res==exp_res
   
@pytest.mark.parametrize("filename",["tests/resources/mitqa/hybridqa/toy.json"])
def test_json_utils(filename):
    data = read_data(filename)
    assert data!=None

row_str = {"player_name":"sachin","score":100}
exp = "player_name is sachin . score is 100 . "
def test_create_table_retriever_training_data():
    res = linearize_row(row_str)
    assert res==exp

def test_evaluate():
    pred = [10,"Joe_Biden",0.78]
    gold = [20,"Joe_Biden",0.78]
    res = normalize_answer(pred[1])
    assert res=="joebiden"
    assert get_tokens(pred[1])==['joebiden']
    assert compute_exact(pred[1],gold[1])==1
    assert compute_f1(pred[1],gold[1])==1.0
    
    
    
    

    

    
