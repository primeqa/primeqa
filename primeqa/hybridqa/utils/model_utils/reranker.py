import json
import numpy as np
import sys
import tqdm
import copy
import os

def get_rc_score(ae_scores,q_id, rank):
    nb = ae_scores[q_id+"_"+str(rank)][0]
    return nb["start_logit"] + nb["end_logit"]

def get_rs_score(rr_scores,q_id, rank):
    return np.array(rr_scores[q_id])[rank]

def re_rank_ae_output(row_retrieval_scores,n_best_prediction_file_path,ae_output_file_path):
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
        # if n != "4":
        #     print(vn["question_id"])
        vn["question_id"] = vn["question_id"].split("_")[0]
        data_rr.append(vn)
    output_file = ae_output_file_path.split(".json")[0]+"_re_ranked.json"
    with open(output_file, "w") as f:
        json.dump(data_rr, f, indent=2)
    # with open(sys.argv[1], "w") as f:
    #     json.dump(data_new, f, indent=2)