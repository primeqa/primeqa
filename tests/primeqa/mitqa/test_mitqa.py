from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from primeqa.mitqa.utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments
import json

from primeqa.mitqa.utils.model_utils.row_retriever_MITQA import RowRetriever
from primeqa.mitqa.utils.model_utils.reranker import re_rank_ae_output
from primeqa.mitqa.utils.model_utils.process_row_retriever_output import preprocess_data_using_row_retrieval_scores,preprocess_data_using_row_retrieval_scores_ottqa,create_dataset_for_answer_extractor
from primeqa.mitqa.utils.model_utils.answer_extractor_multi_Answer import predict_ae
from primeqa.mitqa.processors.preprocessors.preprocess_raw_data import preprocess_data,load_st_model
import logging
import torch
import os
import sys
from primeqa.mitqa.utils.ottqa_utils  import assign_ids

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
        


        
    
        