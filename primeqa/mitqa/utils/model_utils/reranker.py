import json
import numpy as np
import sys
import tqdm
import copy
import os

def get_rc_score(ae_scores,q_id, rank):
    """
    The get_rc_score function takes in the ae_scores dictionary, which is a dictionary of dictionaries.
    The outermost keys are question ids (q_id), and the inner keys are integers from 0 to 10. The values
    are dictionaries with two keys: &quot;start_logit&quot; and &quot;end_logit&quot;. These correspond to the logits for 
    the start and end indices of an answer span, respectively. This function returns the sum of these two 
    values for each rank value up to 10.
    
    Args:
        ae_scores: Get the start and end logits of the answer
        q_id: Identify which question we are looking at
        rank: Select the best answer from the list of candidates
    
    Returns:
        The logit score for the start and end of a span
    """
    nb = ae_scores[q_id+"_"+str(rank)][0]
    return nb["start_logit"] + nb["end_logit"]

def get_rs_score(rr_scores,q_id, rank):
    """
    The get_rs_score function takes in a dictionary of relevance scores,
    a query id, and the rank of a document for that query. It returns the 
    relevance score for that document at that rank.
    
    Args:
        rr_scores: Store the scores of the retrieved documents for each query
        q_id: Retrieve the relevance scores for a given query
        rank: Find the rank of the first relevant document in the ranked list
    
    Returns:
        The relevance score of the document at rank position 'rank' for query with id 'q_id'
    """
    return np.array(rr_scores[q_id])[rank]

def re_rank_ae_output(row_retrieval_scores,n_best_prediction_file_path,ae_output_file_path):
    """
    The re_rank_ae_output function takes in the row retrieval scores, n best prediction file path, and ae output file path.
    It then reranks the ae output by taking into account both row retrieval scores and n best predictions. 
    The function returns an updated version of the ae output with new rerank score for each example.
    
    Args:
        row_retrieval_scores: Get the scores of the retrieved rows
        n_best_prediction_file_path: Specify the path to the n_best_predictions
        ae_output_file_path: Specify the path to the output file of a re-ranked autoencoder
    
    Returns:
        A json file with the re-ranked predictions
    """
    rr_scores = row_retrieval_scores
    ae_scores = json.load(open(n_best_prediction_file_path))
    data = json.load(open(ae_output_file_path))
    data_rr = []
    data_new = []
    mxs = {}
    for example in tqdm.tqdm(data): 
        qas_id = example["question_id"].split("_")[0]
        qas_rank = int(example["question_id"].split("_")[1])
        rcs = get_rc_score(ae_scores,qas_id, qas_rank)
        rss = get_rs_score(rr_scores,qas_id, qas_rank)
        ts = 3.2*rss + rcs
        example["rerank_score"] = ts
        data_new.append(example)
        if qas_id not in mxs.keys() or mxs[qas_id][0] < ts:
            mxs[qas_id] = (ts, example)
    for k, v in tqdm.tqdm(mxs.items()):
        vn = copy.deepcopy(v[1])
        n = vn["question_id"].split("_")[1]
        vn["question_id"] = vn["question_id"].split("_")[0]
        data_rr.append(vn)
    output_file = ae_output_file_path.split(".json")[0]+"_re_ranked.json"
    with open(output_file, "w") as f:
        json.dump(data_rr, f, indent=2)
    return output_file
