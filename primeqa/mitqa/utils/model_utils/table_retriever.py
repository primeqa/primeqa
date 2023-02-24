import os
import argparse, sys
import json
import tempfile
from unittest.mock import patch
from tqdm import tqdm
from primeqa.ir.dense.dpr_top.dpr.biencoder_trainer import BiEncoderTrainer
from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher

def train_table_retriever(root_dir,triples_file_name):
    """
    The train_table_retriever function trains a table retriever model.
    It takes as input the root directory of the dataset and the name of triples file.
    The output is stored in a folder named 'table_retriever' under root directory.
    
    Args:
        root_dir: Specify the directory where all of the files for training are stored
        triples_file_name: Specify the name of the text triples file that will be used for training
    
    Returns:
        The trained model and the tokenizer
    """
    output_dir=os.path.join(root_dir, 'table_retriever')
    os.makedirs(output_dir, exist_ok=True)
    print(output_dir)
    text_triples_fn = os.path.join(root_dir, triples_file_name)

    model_training_args = [
        "prog",
        "--train_dir", text_triples_fn,
        "--output_dir", output_dir,
        "--full_train_batch_size", "256",
        "--num_train_epochs", "3",
        "--training_data_type", "text_triples"]

    with patch.object(sys, 'argv', model_training_args):
        trainer = BiEncoderTrainer()
        trainer.train()
        
def predict_table_retriever(data_path_root,collection_file,raw_data):
    """
    The predict_table_retriever function takes in a data_path_root, collection file and raw data.
    It then creates an output directory for the table retriever model to be stored in. 
    The indexer is called which will create the sharded index of the corpus and save it to output_dir. 
    The searcher is called which will load the qry encoder from model_name_or path and use it to search over all queries in query batch size of 256 on top k = 5 documents from our corpus that are indexed at location specified by index location. The retrieved document IDs are returned along with passages.
    
    Args:
        data_path_root: Specify the path to the root directory of your data
        collection_file: Specify the path to the collection file
        raw_data: Pass the data that we want to predict
    
    Returns:
        A list of dictionaries
    """
    output_dir=os.path.join(data_path_root, 'table_retriever')
    if not os.path.exists(output_dir):
        collection_fn = os.path.join(data_path_root, collection_file)
        indexing_args = [
                "prog",
                "--dpr_ctx_encoder_path", os.path.join(output_dir, "ctx_encoder"),
                "--embed", "1of1",
                "--sharded_index",
                "--batch_size", "256",
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
        query = d['question']
        retrieved_doc_ids, passages = searcher.search(query_batch = [query], top_k = 20, mode = 'query_list')
        for id in range(len(retrieved_doc_ids[0])):
            p_data = {}
            p_data['question'] =query
            p_data['question_id'] = d['question_id']
            p_data["table_id"] = retrieved_doc_ids[0][id]
            p_data["answer-text"] = d['answer-text']
            new_data.append(p_data)
    return new_data