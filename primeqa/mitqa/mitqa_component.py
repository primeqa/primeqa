import json
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
from primeqa.mitqa.utils.model_utils.row_retriever_MITQA import RowRetriever
from primeqa.mitqa.utils.model_utils.reranker import re_rank_ae_output
from primeqa.mitqa.utils.link_predictor import predict_link_for_tables,train_link_generator
from primeqa.mitqa.utils.model_utils.table_retriever import train_table_retriever,predict_table_retriever
from primeqa.mitqa.utils.model_utils.process_row_retriever_output import preprocess_data_using_row_retrieval_scores,create_dataset_for_answer_extractor
from primeqa.mitqa.utils.model_utils.answer_extractor_multi_Answer import train_ae,predict_ae
from primeqa.mitqa.processors.preprocessors.preprocess_raw_data import preprocess_data,load_st_model
import logging
import torch
import os
import sys
from primeqa.mitqa.utils.arguments_utils import HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments
from primeqa.components.base import Component
from primeqa.mitqa.metrics.evaluate_ottqa import get_em_and_f1_ottqa
from primeqa.mitqa.metrics.evaluate import get_em_and_f1_hybridqa

class MITQAReader(Component):
   
   def __init__(self,config_file):
      self._config_file = config_file
   
   def load(self):
      self.logger = logging.getLogger(__name__)
      self.doc_retriever = load_st_model()

      hqa_parser = HfArgumentParser((HybridQAArguments,LinkPredictorArguments, RRArguments,AEArguments))
      self.hqa_args,self.lp_args,self.rr_args,self.ae_args,= hqa_parser.parse_json_file(self._config_file)

   def eval(self):
      pass
   def predict(self):
      """
         Get predictions on the dev/test set of OTTQA/HYBRIDQA datasets.
      """
      self.load()
      raw_test_data = json.load(open(self.hqa_args.test_data_path))
      test=True
      self.ae_args.do_predict_ae = True
      if self.hqa_args.dataset_name=="ottqa":
         retrieved_data = predict_table_retriever(self.hqa_args.data_path_root,self.hqa_args.collections_file,raw_dev_data)
         json.dump(retrieved_data,open(os.path.join(self.hqa_args.data_path_root,"table_retrieval_output_test.json"),"w"))
         linked_data = predict_link_for_tables(self.lp_args,retrieved_data,self.doc_retriever)
         test_data_processed = preprocess_data(self.doc_retriever,self.hqa_args.data_path_root,self.hqa_args.dataset_name,linked_data,split="test",test=test)
      else:
         test_data_processed = preprocess_data(self.doc_retriever,self.hqa_args.data_path_root,self.hqa_args.dataset_name,raw_test_data,split="test",test=test)

      self.logger.info("Initial preprocessing done")
      rr = RowRetriever(self.hqa_args,self.rr_args)
      qid_scores_dict = rr.predict(test_data_processed)
      self.logger.info("Row retrieval predictions Done")
      test_processed_data = preprocess_data_using_row_retrieval_scores(self.doc_retriever,raw_test_data,qid_scores_dict,test)
      self.logger.info("Row retrieval output processed")
      answer_extraction_data = create_dataset_for_answer_extractor(test_processed_data,self.hqa_args.data_path_root,test)
      self.logger.info("Answer extraction data generated")
      ae_output_path,ae_output_path_nbest = predict_ae(self.ae_args,answer_extraction_data)
      self.logger.info(ae_output_path)
      self.logger.info(ae_output_path_nbest)
      re_ranked_output_file = re_rank_ae_output(qid_scores_dict,ae_output_path_nbest,self.ae_args.pred_ans_file) 
      if self.hqa_args.dataset_name=="ottqa":
         self.logger.info(get_em_and_f1_ottqa(re_ranked_output_file,"data/ottqa/released_data/dev_reference.json"))
      else:
         self.logger.info(get_em_and_f1_hybridqa(re_ranked_output_file,"data/ottqa/dev_reference.json"))
      return re_ranked_output_file
   
   def train(self):
      """
         Train the model on OTTQA/HYBRIDQA train set and evaluate on dev set and repot EM and F1 scores on dev set.
      """
      self.load()
      test =False
      raw_train_data = json.load(open(self.hqa_args.train_data_path))
      raw_dev_data = json.load(open(self.hqa_args.dev_data_path))
      if self.hqa_args.dataset_name == "ottqa":
         if self.hqa_args.train_tr:
            train_table_retriever(self.hqa_args.data_path_root,"triples_train.tsv")
         retrieved_data_train = predict_table_retriever(self.hqa_args.data_path_root,self.hqa_args.collections_file,raw_train_data)
         json.dump(retrieved_data_train,open(os.path.join(self.hqa_args.data_path_root,"table_retrieval_output_train.json"),"w"))
         if self.hqa_args.train_lp:
            train_link_generator(self.lp_args)
         linked_data_train = predict_link_for_tables(self.lp_args,retrieved_data_train,self.doc_retriever)
         retrieved_data_dev = predict_table_retriever(self.hqa_args.data_path_root,self.hqa_args.collections_file,raw_dev_data)
         json.dump(retrieved_data_dev,open(os.path.join(self.hqa_args.data_path_root,"table_retrieval_output_dev.json"),"w"))
         linked_data_dev = predict_link_for_tables(self.self.lp_args,retrieved_data_dev)
         train_data_processed = preprocess_data(self.doc_retriever,self.hqa_args.data_path_root,self.hqa_args.dataset_name,linked_data_train,split="train",test=test)
         dev_data_processed = preprocess_data(self.doc_retriever,self.hqa_args.data_path_root,self.hqa_args.dataset_name,linked_data_dev,split="dev",test=test)
      else:
         train_data_processed = preprocess_data(self.doc_retriever,self.hqa_args.data_path_root,self.hqa_args.dataset_name,raw_train_data,split="train",test=test)
         dev_data_processed = preprocess_data(self.doc_retriever,self.hqa_args.data_path_root,self.hqa_args.dataset_name,raw_dev_data,split="dev",test=test)
      self.logger.info("Train: Initial preprocessing done")
      rr = RowRetriever(self.hqa_args,self.rr_args)
      self.logger.info("Train: Training row retrieval model")
      rr.train(train_data_processed,dev_data_processed)
      qid_scores_dict_train = rr.predict(train_data_processed)
      qid_scores_dict_dev = rr.predict(dev_data_processed)
      train_processed_data = preprocess_data_using_row_retrieval_scores(self.doc_retriever,raw_train_data,qid_scores_dict_train,test)
      dev_processed_data = preprocess_data_using_row_retrieval_scores(self.doc_retriever,raw_dev_data,qid_scores_dict_dev,test)
      answer_extraction_train_data = create_dataset_for_answer_extractor(train_processed_data,self.hqa_args.data_path_root,test)
      answer_extraction_dev_data = create_dataset_for_answer_extractor(dev_processed_data,self.hqa_args.data_path_root,test)
      output_dir = train_ae(self.ae_args,answer_extraction_train_data)
      ae_output_path,ae_output_path_nbest = predict_ae(self.ae_args,answer_extraction_dev_data)
      re_ranked_output_file = re_rank_ae_output(qid_scores_dict_dev,ae_output_path_nbest,self.ae_args.pred_ans_file) 
      self.logger.info(f"Train: Training Done model saved at: {output_dir}")
      if self.hqa_args.dataset_name=="ottqa":
         self.logger.info(get_em_and_f1_ottqa(re_ranked_output_file,"data/ottqa/released_data/dev_reference.json"))
      else:
         self.logger.info(get_em_and_f1_hybridqa(re_ranked_output_file,"data/ottqa/dev_reference.json"))
   
      
      
   
      
  