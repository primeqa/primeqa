import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch
from tqdm import tqdm
import numpy as np
import json
from primeqa.hybridqa.utils.table_utils import fetch_table
import sys
import argparse


if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


def linearize(row):
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" is "+str(r)+" . "
        # row_str+=str(c)+" "+str(r)+" "
    return row_str

def create_dataset_for_answer_extractor(data, data_path_root,test=False):
    #print(len(data))
    label_1_data = []
    prev_qid = ""
    i=1
    no_found = 0
    found_set = set([])
    output_file = os.path.join(data_path_root,"ae_input_train.json") if test else os.path.join(data_path_root,"ae_input_test.json")
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
model_name = 'msmarco-distilbert-base-tas-b'
doc_retriever = SentenceTransformer(model_name)
top_k = 2

def get_top_k_passages(passages,query,top_k, row=None):
    old_passages = passages
    if row is not None:
        row_str = ""
        for k, v in row.items():
            row_str += " " + k + " is " + v + " . "
        passages = [row_str+passage for passage in passages]
    # print(passages)
    corpus_embeddings = doc_retriever.encode(passages, convert_to_tensor=True, show_progress_bar=False)
    # print(corpus_embeddings)
    corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = doc_retriever.encode([query], convert_to_tensor=True)
    # print(query_embeddings)
    query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
    hits = hits[0]
    #print(hits)
    relevant_sents =[]
    for hit in hits:
        #print("\t{:.3f}\t{}".format(hit['score'], passage[hit['corpus_id']]))
        relevant_sents.append(old_passages[hit['corpus_id']])
    return relevant_sents

def get_max_score_row(p,q_id):
    return np.array(p[q_id]).argsort()[-5:]

def preprocess_instance(d,test=False):
    p_d = {}
    p_d['question'] = d['question']
    p_d['question_id'] = d['question_id']
    if not test:
        p_d['answer-text'] = d['answer-text']
    p_d['table'] = fetch_table(d['table_id'])
    p_d['table_id'] = d['table_id']
    return p_d
    

def preprocess_data_using_row_retrieval_scores(raw_data,qid_scores_dict,test):
    #data = json.load(open(data_path))
    #p = json.load(open(row_ret_pred_path))
    p = qid_scores_dict
    print(p)
    processed_data = []
    num = 0
    den = 0
    for d in tqdm(raw_data):
        # if d['label'] != 1:
        #     continue
        pi = preprocess_instance(d,test=test)
        question_str = pi['question']
        if not test:
            answer_text = pi['answer-text']
        q_id = pi['question_id']
        table_id = pi['table_id']
        
        #question = [d['question']]
        #q_input = self.tokenizer(question,add_special_tokens=True, truncation=True,padding=True, return_tensors='pt', max_length = self.max_seq_len)
        header = pi['table']['header']
        cri = get_max_score_row(p,q_id) #d['correct_row_index']
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
            # l1, l2 = sorted([v.strip().lower() for v in one_row.values()]), sorted([v.strip().lower() for v in d['table_row'].values()])
            # xl.append([l1, l2])# print(l1)
            # # print(l2)
            # if  l1 != l2:
            #     continue
            # nm += 1
            table_rows.append(one_row)
            table_row_passages.append(passage)
            table_row_passages_new.append(passages)
        # if (nm==0):
        #     print(xl)
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
                #npr = " ".join(npr)

            
            npi['table_passage_row'] = npr
            
            if not test:
                npi['answer-text'] = answer_text
                # if answer_text.lower() in pr.lower() or answer_text.lower() in row_values:
                npi['label'] =1

                if answer_text.lower() in npr.lower() or answer_text.lower() in row_values:
                    npi['label_new'] = 1
                else:
                    npi['label_new'] = 0
                
                if (npi['label_new']!=npi['label']):
                    num += 1
                den += 1

                # else:
                #     npi['label'] = 1
            

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
