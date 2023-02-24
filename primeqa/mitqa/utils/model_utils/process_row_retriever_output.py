import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from tqdm import tqdm
import numpy as np
import json
from primeqa.mitqa.utils.table_utils import fetch_table
import sys
import argparse


if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


def linearize(row):
    """
    The linearize function takes a row of the table and returns a string
    representation of that row. The string representation is just each column
    name followed by its value, separated by spaces, with every word separated
    by spaces as well. For example: "player is Sachin . "
    
    Args:
        row: A table row
    
    Returns:
        A string of the form: "column header is value ."
    """
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" is "+str(r)+" . "
    return row_str

def create_dataset_for_answer_extractor(data, data_path_root,test=False):
    """
    The create_dataset_for_answer_extractor function takes in a list of dictionaries, each dictionary representing
    a row from the original data file. Each dictionary contains keys for 'question_id', 'table_id', and 'answer-text'.
    The function then iterates through this list of dictionaries, and for each one that has an answer label of 1 (i.e., 
    the correct answer is present in the table), it extracts the context (i.e., all rows concatenated together) from 
    the corresponding table, as well as the question text itself. These are stored under keys &quot;context&quot; and &quot;question&quot;, respectively.
    
    Args:
        data: Create the dataset
        data_path_root: Specify the path where the data is stored
        test: Create the test dataset
    
    Returns:
        A list of dictionaries
    """
    #print(len(data))
    label_1_data = []
    prev_qid = ""
    i=1
    no_found = 0
    found_set = set([])
    output_file = os.path.join(data_path_root,"ae_input_test.json") if test else os.path.join(data_path_root,"ae_input_train.json")
    for d in tqdm(data):
        new_data = {}
        if test or d['label'] == 1: # or d['match_score'] != '-INF':
            if not test:
                orig_answer = d['answer-text']
                new_data['answer-text'] = orig_answer
            
            context = linearize(d['table_row']) + d['table_passage_row']
            new_data['context'] = context
            new_data['title'] = d['table_id'].replace("_"," ")
            new_data['question'] = d['question']
            if prev_qid ==d['question_id']:
                new_qid = d['question_id']+"_"+str(i)
                prev_qid = d['question_id']
                i+=1
            else:
                i=1
                new_qid = d['question_id']+"_"+str(0)
                prev_qid = d['question_id']
            new_data['question_id'] = new_qid
            if not test:
                start = context.lower().find(orig_answer.lower())
                if start == -1:
                    # print(context, orig_answer)
                    no_found += 1
                    # import pdb
                    # pdb.set_trace()
                    answer = "NOT FOUND"
                    new_data['is_impossible'] = True
                else:
                    new_data['is_impossible'] = False
                    found_set.add(d['question_id'])
                    start = context.lower().find(orig_answer.lower())
                    assert(start!=-1)
                    while context[start].lower() != orig_answer[0].lower():
                        start -= 1
                    answer = context[start:start+len(orig_answer)]
                new_data['answers'] = [{'answer_start': start, 'text': answer}]
            label_1_data.append(new_data)

    print("total", len(label_1_data), "answer not found in", no_found, "found in", len(found_set))
    json.dump(label_1_data,open(output_file,"w"), indent=4)
    return label_1_data


# We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
top_k = 2

def get_top_k_passages(doc_retriever,passages,query,top_k, row=None):
    """
    The get_top_k_passages function takes in a doc_retriever object, a list of passages (strings),
    a query string, and an integer k. It returns the top k passages that are most relevant to the query.
    
    
    Args:
        doc_retriever: Encode the passages and query
        passages: Pass in the passages that are being used to search for the query
        query: Search for the relevant passages
        top_k: Specify how many of the top passages to return
        row: Pass the current row of the dataframe to get_top_k_passages
    
    Returns:
        The top k passages based on the semantic similarity of the query and each passage
    """
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

def get_max_score_row(p,q_id):
    """
    The get_max_score_row function takes in a list of lists (p) and an integer (q_id).
    It returns the indices of the 5 rows with the highest scores for query q_id.
    
    
    Args:
        p: Store the scores of each question
        q_id: Select the row of p that corresponds to the question
    
    Returns:
        The row numbers of the top 5 scores for a given query
    """
    return np.array(p[q_id]).argsort()[-1:]

def preprocess_instance(d,test=False):
    p_d = {}
    p_d['question'] = d['question']
    p_d['question_id'] = d['question_id']
    if not test:
        p_d['answer-text'] = d['answer-text']
    p_d['table'] = fetch_table(d['table_id'])
    p_d['table_id'] = d['table_id']
    return p_d

def preprocess_ottqa_instance(d,test=False):
    p_d = {}
    p_d['question'] = d['question']
    p_d['question_id'] = d['question_id']
    if not test:
        p_d['answer-text'] = d['answer-text']
    p_d['table_id'] = d['table_id']
    return p_d
    
def preprocess_data_using_row_retrieval_scores_ottqa(raw_dataset_with_ids,qid_scores_dict,test):
    """
    The preprocess_data_using_row_retrieval_scores_ottqa function takes in a dataset with question_ids and the qid_scores dictionary. It then uses the scores to filter out rows that have low retrieval scores for each question. The function returns a new dataset with only those questions that had high retrieval scores.
    
    Args:
        raw_dataset_with_ids: Pass the raw dataset with ids
        qid_scores_dict: Store the scores of each question
        test: Decide whether the function is used for training or testing
    
    Returns:
        A list of dictionaries with the question_id replaced by the prefix_qid
    """
    
    p = qid_scores_dict
    new_data =[]
    for d in tqdm(raw_dataset_with_ids):
        question_id = d['question_id']
        prefix_qid = question_id.split("_")[0]
        suffix_qid = question_id.split("_")[1]
        topk = get_max_score_row(p,prefix_qid)
        if int(suffix_qid) in topk:
            d['question_id']=prefix_qid
            new_data.append(d)
        else:
            continue 
    return new_data

    
    


def preprocess_data_using_row_retrieval_scores(doc_retriever,raw_data,qid_scores_dict,test):
    """
    The preprocess_data_using_row_retrieval_scores function takes in a list of dictionaries, each dictionary representing an instance.
    The function then preprocesses the data by retrieving the top 100 passages for each question and table pair. 
    It then creates a new dictionary with all of these information as keys and values. The function returns this list of dictionaries.
    
    Args:
        doc_retriever: Retrieve the passages for each row
        raw_data: Pass the data to be preprocessed
        qid_scores_dict: Store the scores of each question-id
        test: Determine whether the data is being processed for training or testing
    
    Returns:
        The processed data
    """
    p = qid_scores_dict
    processed_data = []
    num = 0
    den = 0
    for d in tqdm(raw_data):
        pi = preprocess_instance(d,test=test)
        question_str = pi['question']
        if not test:
            answer_text = pi['answer-text']
        q_id = pi['question_id']
        table_id = pi['table_id']
        header = pi['table']['header']
        cri = get_max_score_row(p,q_id)
        rows =  [pi['table']['data'][cr] for cr in cri]
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
                npr = " ".join(get_top_k_passages(doc_retriever,npr, question_str, 100, r))

            
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
    return processed_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--released_data_path", type=str, default="data/released_data/test.json",
                    help="Path to processed data")
    parser.add_argument("--row_ret_pred_path", type=str, default="data/predictions/row_retriever/on_test_BLarge_no_group_ranked_passage.json",
                    help="Path to row retrieval prediction file")
    parser.add_argument("--processed_output_file_path", type=str, default="data/processed_data/input_for_answer_extractor.json",
                    help="Path to processed output file which will be fed as input to answer extractor for prediction")
    parser.add_argument("--test", help="Test set",
                    action="store_true")

    args = parser.parse_args()

    test=True
    processed_data = preprocess_data_using_row_retrieval_scores(args.released_data_path,test,args.row_ret_pred_path)
    create_dataset_for_answer_extractor(processed_data,args.processed_output_file_path,test)
