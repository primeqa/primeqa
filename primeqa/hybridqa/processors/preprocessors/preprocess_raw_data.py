import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from tqdm import tqdm

import json
from utils.table_utils import fetch_table
import sys

if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


model_name = 'msmarco-distilbert-base-tas-b' 
doc_retriever = SentenceTransformer(model_name)

""" Rank passages most relevant to the query higher than other passages """
def get_top_k_passages(passages,query,top_k, row=None):
    old_passages = passages
    if row is not None:
        row_str = ""
        for k, v in row.items():
            row_str += " " + k + " is " + v + " . "
        passages = [row_str+passage for passage in passages]
    corpus_embeddings = doc_retriever.encode(passages, convert_to_tensor=True, show_progress_bar=False)
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = doc_retriever.encode([query], convert_to_tensor=True)
    query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
    hits = hits[0]
    relevant_sents =[]
    for hit in hits:
        relevant_sents.append(old_passages[hit['corpus_id']])
    return relevant_sents


""" Preprocess a single instance """
def preprocess_instance(d,test=False):
    p_d = {}
    p_d['question'] = d['question']
    p_d['question_id'] = d['question_id']
    if not test:
        p_d['answer-text'] = d['answer-text']
    p_d['table'] = fetch_table(d['table_id'])
    p_d['table_id'] = d['table_id']
    return p_d




""" Preprocess the full data """
def preprocess_data(data_root_path,data_path,split,test):
    data = json.load(open(data_path))
    processed_data_path = os.path.join(data_root_path,str(split)+"_processed.json")
    processed_data = []
    num = 0
    den = 0
    for d in tqdm(data):
        # if d['label'] != 1:
        #     continue
        pi = preprocess_instance(d,test=test)
        question_str = pi['question']
        if not test:
            answer_text = pi['answer-text']
        q_id = pi['question_id']
        table_id = pi['table_id']
        header = pi['table']['header']
        rows =  [pi['table']['data'][i] for i in range(len(pi['table']['data']))]
        table_rows = []
        table_row_passages = []
        table_row_passages_new = []
        nm, xl = 0, []
        for r in rows:
            one_row = {}
            passage = ""
            passages = []
            for r_v,h in zip(r,header):
                one_row[h] = r_v["cell_value"]
                passage+= " ".join(r_v['passages'])
                passages += r_v['passages']
            table_rows.append(one_row)
            table_row_passages.append(passage)
            table_row_passages_new.append(passages)

        for r,pr,npr in zip(table_rows,table_row_passages,table_row_passages_new):
            npi={}
            npi['question_id'] = q_id
            npi['question'] = question_str
            npi['table_id'] = table_id
            npi['table_row'] = r
            row_values = [v.lower() for v in r.values()]
            npi['table_passage_row_old'] = pr
            npi['table_passage_row'] = pr

            if (len(npr)==0):
                npr = ""
            elif (len(npr)==1):
                npr = npr[0]
            else:
                npr = " ".join(get_top_k_passages(npr, question_str, 100, r))
            
            npi['table_passage_row'] = npr
            
            if not test:
                npi['answer-text'] = answer_text
                npi['label'] =1

                if answer_text.lower() in npr.lower() or answer_text.lower() in row_values:
                    npi['label_new'] = 1
                else:
                    npi['label_new'] = 0
                
                if (npi['label_new']!=npi['label']):
                    num += 1
                den += 1

            processed_data.append(npi)
    print("total", den, "changed", num, len(processed_data))
    json.dump(processed_data,open(processed_data_path,"w"),indent=4)
    return processed_data


if __name__ == "__main__":
    rel_data_path = sys.argv[1] # Released data path 
    processed_data_path = sys.argv[2] # Output file path
    processed_data = preprocess_data(rel_data_path,True)

    json.dump(processed_data,open(processed_data_path,"w"),indent=4)
