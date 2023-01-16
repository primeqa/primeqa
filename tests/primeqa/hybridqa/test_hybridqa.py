from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from primeqa.hybridqa.utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments
import json

from primeqa.hybridqa.utils.model_utils.row_retriever_MITQA import RowRetriever
from primeqa.hybridqa.utils.model_utils.reranker import re_rank_ae_output
from primeqa.hybridqa.utils.model_utils.process_row_retriever_output import preprocess_data_using_row_retrieval_scores,preprocess_data_using_row_retrieval_scores_ottqa,create_dataset_for_answer_extractor
from primeqa.hybridqa.utils.model_utils.answer_extractor_multi_Answer import predict_ae
from primeqa.hybridqa.processors.preprocessors.preprocess_raw_data import preprocess_data,load_st_model
import logging
import torch
import os
import sys
from primeqa.hybridqa.utils.ottqa_utils  import assign_ids

from primeqa.hybridqa.utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments
from primeqa.hybridqa.utils.model_utils.table_retriever import train_table_retriever,predict_table_retriever
from primeqa.hybridqa.utils.link_predictor import predict_link_for_tables,train_link_generator


hybridqa_config = {
    "per_device_train_batch_size_rr":8,
    "per_device_eval_batch_size_rr":8,
    "rr_model_name":"bert-base-uncased",
    "row_retriever_model_name_path":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/pretrained_models/rr.bin",
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
    "pred_ans_file":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/predictions/answer_extractor_output_test.json",
    "eval_file":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/ae_input_test.json",
    "output_dir":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/models/answer_extractor/",
    "model":"gpt2",
    "top_k":0,
    "top_p":0.9,
    "seed_lg":42,
    "batch_size_lg":2,
    "linker_model":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/ottqa/models/link_generator/model-ep9.pt",
    "max_source_len":32,
    "max_target_len":16,
    "do_all_lg":True,
    "data_path_root":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/",
    "dataset_name":"hybridqa",
    "test_data_path":"/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/toy.json",
    "collections_file":"linearized_tables.tsv",
    "test":True
}

ottqa_config = {
    "per_device_train_batch_size_rr":32,
    "per_device_eval_batch_size_rr":32,
    "rr_model_name":"bert-large-uncased",
    "row_retriever_model_name_path":"data/hybridqa/models/row_retriever/Large_no_group_ranked_passage.bin",
    "pos_frac_per_epoch":[0.51, 0.5, 0.3, 0.1, 0.09],
    "group_frac_per_epoch":[0.0, 0.0, 0.0, 0.0, 0.0],
    "max_seq_length":512,
    "per_gpu_train_batch_size":16,
    "train_batch_size":8,
    "per_gpu_eval_batch_size":16,
    "eval_batch_size":8,
    "max_query_length":64,
    "threads":1,
    "null_score_diff_threshold":0.0,
    "n_best_size":20,
    "do_predict_ae":True,
    "n_gpu":1,
    "max_answer_length":30,
    "model_name_or_path_ae":"bert-large-uncased-whole-word-masking-finetuned-squad",
    "output_dir":"data/ottqa/models/answer_extractor/",
    "model_type":"bert",
    "doc_stride":128,
    "pred_ans_file":"data/ottqa/predictions/answer_extractor_output_test.json",
    "eval_file":"data/ottqa/ae_input_test.json",
    "model":"gpt2",
    "top_k":0,
    "top_p":0.9,
    "seed_lg":42,
    "batch_size_lg":2,
    "linker_model":"data/ottqa/models/link_generator/model-ep9.pt",
    "max_source_len":32,
    "max_target_len":16,
    "do_all_lg":True,
    "data_path_root":"data/ottqa/",
    "dataset_name":"ottqa",
    "test_data_path":"data/ottqa/released_data/dev.json",
    "collections_file":"linearized_tables.tsv",
    "test":True
}

class TestHybridqa():
    def test(self):
        hqa_parser = HfArgumentParser((HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments))
        test=True
        hqa_args,lp_args,rr_args,ae_args,= hqa_parser.parse_dict(hybridqa_config)
        raw_test_data = json.load(open(hqa_args.test_data_path))
        doc_retriever = load_st_model()
        assert len(raw_test_data) !=0
        test_data_processed = preprocess_data(doc_retriever,hqa_args.data_path_root,hqa_args.dataset_name,raw_test_data,split="test",test=test)
        assert test_data_processed[0]['table_passage_row'] != None
        assert len(raw_test_data) == len(test_data_processed)
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
        
        
class TestOTTQA():
    def test_ottqa(self):
        hqa_parser = HfArgumentParser((HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments))
        hqa_args,lp_args,rr_args,ae_args,= hqa_parser.parse_dict(hybridqa_config)
        raw_test_data = json.load(open(hqa_args.test_data_path))
        doc_retriever = load_st_model()
        assert len(raw_test_data) !=0
        retrieved_data = predict_table_retriever(hqa_args.data_path_root,hqa_args.collections_file,raw_test_data)
        assert(len(retrieved_data)!=0)
        linked_data = predict_link_for_tables(lp_args,retrieved_data,doc_retriever)
        test_data_processed = preprocess_data(doc_retriever,hqa_args.data_path_root,hqa_args.dataset_name,linked_data,split="test",test=test)
        assert len(raw_test_data) == len(test_data_processed)
        rr = RowRetriever(hqa_args,rr_args)
        qid_scores_dict = rr.predict(test_data_processed)
        assert qid_scores_dict != None
        raw_test_data_with_ids = assign_ids(test_data_processed)
        test_processed_data = preprocess_data_using_row_retrieval_scores_ottqa(raw_test_data_with_ids,qid_scores_dict,test)
        assert len(test_processed_data) == len(test_data_processed)
        answer_extraction_data = create_dataset_for_answer_extractor(test_processed_data,hqa_args.data_path_root,test)
        assert answer_extraction_data != None
        ae_output_path,ae_output_path_nbest = predict_ae(ae_args,answer_extraction_data)
        re_ranked_output = re_rank_ae_output(qid_scores_dict,ae_output_path_nbest,ae_args.pred_ans_file) 
        assert re_ranked_output!= None


        


        
    
        