import json
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from utils.model_utils.row_retriever_MITQA import RowRetriever
from utils.model_utils.reranker import re_rank_ae_output
from utils.link_predictor import predict_link_for_tables,train_link_generator
from utils.model_utils.table_retriever import train_table_retriever,predict_table_retriever
from utils.model_utils.process_row_retriever_output import preprocess_data_using_row_retrieval_scores,create_dataset_for_answer_extractor
from utils.model_utils.answer_extractor_multi_Answer import run_answer_extractor
from processors.preprocessors.preprocess_raw_data import preprocess_data
import logging
import torch
import os
import sys
from utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments


def run_hybrid_qa():
   logger = logging.getLogger(__name__)
   logger.info("running hybridqa")
   hqa_parser = HfArgumentParser((HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments))

   if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
      hqa_args,lp_args,rr_args,ae_args,= hqa_parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
   else:
      hqa_args,lp_args,rr_args,ae_args, = hqa_parser.parse_args_into_dataclasses()
   logger.info(hqa_args,lp_args,rr_args,ae_args)
   raw_train_data = json.load(open(hqa_args.train_data_path))
   raw_dev_data = json.load(open(hqa_args.dev_data_path))
   raw_test_data = json.load(open(hqa_args.test_data_path))
   test=False
   if hqa_args.test_data_path is not None and hqa_args.test:
      logger.info("Test Mode")
      test=True
      ae_args.do_predict_ae = True
      if hqa_args.dataset_name=="ottqa":
         retrieved_data = predict_table_retriever(hqa_args.data_path_root,hqa_args.collections_file,raw_dev_data)
         json.dump(retrieved_data,open(os.path.join(hqa_args.data_path_root,"table_retrieval_output_test.json"),"w"))
         linked_data = predict_link_for_tables(lp_args,retrieved_data)
         test_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dataset_name,linked_data,split="test",test=test)
      else:
         test_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dataset_name,raw_test_data,split="test",test=test)

      logger.info("Initial preprocessing done")
      rr = RowRetriever(hqa_args,rr_args)
      qid_scores_dict = rr.predict(test_data_processed)
      logger.info("Row retrieval predictions Done")
      test_processed_data = preprocess_data_using_row_retrieval_scores(raw_dev_data,qid_scores_dict,test)
      logger.info("Row retrieval output processed")
      answer_extraction_data = create_dataset_for_answer_extractor(test_processed_data,hqa_args.data_path_root,test)
      logger.info("Answer extraction data generated")
      ae_output_path,ae_output_path_nbest = run_answer_extractor(ae_args,answer_extraction_data)
      logger.info(ae_output_path)
      logger.info(ae_output_path_nbest)
      re_rank_ae_output(qid_scores_dict,ae_output_path_nbest,ae_args.pred_ans_file) 
   else:
      logger.info("Training Mode")
      if hqa_args.dataset_name == "ottqa":
         if hqa_args.train_tr:
            train_table_retriever(hqa_args.data_path_root,"triples_train.tsv")
         retrieved_data_train = predict_table_retriever(hqa_args.data_path_root,hqa_args.collections_file,raw_train_data)
         json.dump(retrieved_data_train,open(os.path.join(hqa_args.data_path_root,"table_retrieval_output_train.json"),"w"))
         if hqa_args.train_lp:
            train_link_generator(lp_args)
         linked_data_train = predict_link_for_tables(lp_args,retrieved_data_train)
         retrieved_data_dev = predict_table_retriever(hqa_args.data_path_root,hqa_args.collections_file,raw_dev_data)
         json.dump(retrieved_data_dev,open(os.path.join(hqa_args.data_path_root,"table_retrieval_output_dev.json"),"w"))
         linked_data_dev = predict_link_for_tables(lp_args,retrieved_data_dev)
         train_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dataset_name,linked_data_train,split="train",test=test)
         dev_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dataset_name,linked_data_dev,split="dev",test=test)
      else:
         train_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dataset_name,raw_train_data,split="train",test=test)
         dev_data_processed = preprocess_data(hqa_args.data_path_root,hqa_args.dataset_name,raw_dev_data,split="dev",test=test)
      logger.info("Train: Initial preprocessing done")
      rr = RowRetriever(hqa_args,rr_args)
      logger.info("Train: Training row retrieval model")
      rr.train(train_data_processed,dev_data_processed)
      qid_scores_dict_train = rr.predict(train_data_processed)
      qid_scores_dict_dev = rr.predict(dev_data_processed)
      train_processed_data = preprocess_data_using_row_retrieval_scores(raw_train_data,qid_scores_dict_train,test)
      dev_processed_data = preprocess_data_using_row_retrieval_scores(raw_dev_data,qid_scores_dict_dev,test)
      answer_extraction_train_data = create_dataset_for_answer_extractor(train_processed_data,hqa_args.data_path_root,test)
      answer_extraction_dev_data = create_dataset_for_answer_extractor(dev_processed_data,hqa_args.data_path_root,test)
      output_dir = run_answer_extractor(ae_args,answer_extraction_train_data)
      ae_args.do_train_ae = False
      ae_output_path,ae_output_path_nbest = run_answer_extractor(ae_args,answer_extraction_dev_data)
      re_rank_ae_output(qid_scores_dict_dev,ae_output_path_nbest,ae_args.pred_ans_file) 
      logger.info(f"Train: Training Done model saved at: {output_dir}")
      
      
      
      
      
if __name__ == '__main__':
    run_hybrid_qa()