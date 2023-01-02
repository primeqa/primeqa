import os
import argparse, sys

import tempfile
from unittest.mock import patch
from tqdm import tqdm
from primeqa.ir.dense.dpr_top.dpr.biencoder_trainer import BiEncoderTrainer
from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher

def train_table_retriever(root_dir,triples_file_name):
    #test_files_location = 'data/ottqa/'
    output_dir=os.path.join(root_dir, 'table_retriever')
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    text_triples_fn = os.path.join(root_dir, triples_file_name)

    model_training_args = [
        "prog",
        "--train_dir", text_triples_fn,
        "--output_dir", output_dir,
        "--full_train_batch_size", "1",
        "--num_train_epochs", "1",
        "--training_data_type", "text_triples"]

    with patch.object(sys, 'argv', model_training_args):
        trainer = BiEncoderTrainer()
        trainer.train()
        
def predict_table_retriever(data_path_root,collection_file,raw_data):
    output_dir=os.path.join(data_path_root, 'table_retriever')
    if not os.path.exists(output_dir):
        collection_fn = os.path.join(data_path_root, collection_file)
        indexing_args = [
                "prog",
                "--dpr_ctx_encoder_path", os.path.join(output_dir, "ctx_encoder"),
                "--embed", "1of1",
                "--sharded_index",
                "--batch_size", "1",
                "--corpus", collection_fn,
                "--output_dir", output_dir]    
        with patch.object(sys, 'argv', indexing_args):
            indexer = DPRIndexer()
            indexer.index()
    
    search_args = [
    "prog",
    "--model_name_or_path", os.path.join(output_dir, "qry_encoder"),
    "--index_location", output_dir,
    "--output_dir", output_dir]  

    with patch.object(sys, 'argv', search_args):
        searcher = DPRSearcher()
    new_data = []
    for d in tqdm(raw_data):
        p_data = {}
        query = d['question']
        retrieved_doc_ids, passages = searcher.search(query_batch = [query], top_k = 1, mode = 'query_list')
        p_data['question'] =query
        p_data['question_id'] = d['question_id']
        p_data["table_id"] = retrieved_doc_ids[0][0]
        p_data["answer-text"] = d['answer-text']
        new_data.append(p_data)
    return new_data
        
if __name__=="__main__":
    train_table_retriever()   
