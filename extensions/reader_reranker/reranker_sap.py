
# load result
from glob import glob
import json
import pandas as pd
import ast
import numpy as np


# sort by confidence score
def rank_by_confidence(results, data, alpha=.7, beta=.3):
    
    ranked_answers = []
    order_answers = []

    for qid in results:
        seen_ids = set()
        answer_per_passage = []
        # passage_scores = ast.literal_eval(data.iloc[int(qid)]['passage_scores'])
        passage_scores = data.iloc[int(qid)]['normalized_passage_scores']
        pids = ast.literal_eval(data.iloc[int(qid)]['document_ids'])
        for answer in results[qid]['answers']:
            answer['passage_score'] = passage_scores[pids.index(answer['passage_index'])]
            seen_ids.add(answer['passage_index'])
            answer_per_passage.append(answer)
        
        for pid in pids:
            if pid not in seen_ids:
                answer = {}
                answer['passage_index'] = pid
                answer['passage_score'] = passage_scores[pids.index(pid)]
                answer['cls_score'] = 0
                answer['span_answer_score'] = 0
                answer['confidence_score'] = 0
                answer['example_id'] = qid
                answer_per_passage.append(answer)
        ranked = sorted(answer_per_passage, key=lambda x:(x['example_id'],[alpha*x['passage_score']+beta*x['confidence_score'],x['passage_score']]), reverse=True)

        order = [len(pids)]*(len(pids))
        index = 0
        for answer in ranked:
            if order[pids.index(answer['passage_index'])] == len(pids): 
               order[pids.index(answer['passage_index'])] = index
               index += 1
            if index >= len(pids):
                break
        ranked_answers.append(ranked)
        order_answers.append(order)
    return ranked_answers, order_answers

# compute original and new recall
# def compute_recall(data, order_answers):
#     # R@ 1, 3, 10, 40
#     recall_counts = [1, 3, 10, 40]
#     orig_scores = {'1':0,'3':0,'10':0,'40':0}
#     rerank_scores = {'1':0,'3':0,'10':0,'40':0}

#     # check rank of gold id
#     for i, row in data.iterrows():
#         best_original = 100
#         best_rerank = 100
#         for gold_doc_id in ast.literal_eval(row['gold_document_ids']):
#             document_ids = ast.literal_eval(row['document_ids'])
#             if gold_doc_id not in document_ids:
#                 continue
#             if 'order' in row:
#                 if document_ids.index(gold_doc_id) <= len(ast.literal_eval(row['order'])):
#                     original = ast.literal_eval(row['order'])[document_ids.index(gold_doc_id)]
#             else:
#                 original = document_ids.index(gold_doc_id)
#             if document_ids.index(gold_doc_id) <= len(order_answers[i]):
#                 rerank = order_answers[i][(document_ids.index(gold_doc_id))]
#             if original < best_original:
#                 best_original = original
#             if rerank < best_rerank:
#                 best_rerank = rerank
        
#         for r_count in recall_counts:
#             if best_original <= r_count:
#                 orig_scores[str(r_count)] += 1
#             if best_rerank <= r_count:
#                 rerank_scores[str(r_count)] += 1

#     for score in orig_scores:
#         orig_scores[score] = orig_scores[score]/len(data)
#     for score in rerank_scores:
#         rerank_scores[score] = rerank_scores[score]/len(data)

#     return orig_scores, rerank_scores

def doc_match_at_k(row, k_params=[1, 3, 10]):


    # update order of document_ids to match ['order']
    new_document_ids = [None]*len(ast.literal_eval(row['document_ids']))

    index = 0
    for position in row['order']:
        new_document_ids[position] = ast.literal_eval(row['document_ids'])[index]
        index += 1

    gold_doc_ids = ast.literal_eval(row['gold_document_ids'])

    # # match only the first passage if requested to
    # if not metric_params['match_any_gold_passage']:
    #     gold_doc_ids = gold_doc_ids[:1]

    gold_doc_ids = set(gold_doc_ids)
    pred_doc_ids = new_document_ids #list(row['document_ids'])
    k_values = []
    match = False
    for k in k_params:
        if not match:
            match = len(set(pred_doc_ids[:k]).intersection(gold_doc_ids)) > 0
        k_values.append(match)

    return pd.Series(k_values, index=k_params)

def normalize_passage_scores(data):

    normalized_scores = []
    for i, row in data.iterrows():
        passage_scores = np.array(ast.literal_eval(row['passage_scores']))
        passage_scores_norm = (passage_scores-np.min(passage_scores))/(np.max(passage_scores)-np.min(passage_scores))
        normalized_scores.append(passage_scores_norm.tolist())
    data['normalized_passage_scores'] = normalized_scores

# compute original and new recall
def update_data(data, order_answers):

    data['order'] = order_answers
    return data
    # # check rank of gold id
    # for i, row in data.iterrows():
    #     if 'order' in data.columns:
    #         data.iloc[0,data.columns.get_loc('order')] = order_answers[i]
    #     # else:
    #         # data.iloc[0]['order'] = str(order_answers[i])
    # return data            

data_df = pd.read_csv("/dccstor/srosent3/reranking/mf-coga/experiments/sap_reranking/output/dataset-retrieval-reranking=none" + "/output.csv", header=0)

files = glob("/dccstor/srosent3/reranking/mf-coga/experiments/sap_reranking/output/dataset-retrieval-reranking=none/reader_answers.json")
files.sort()
print(files)

results = {}
for file_name in files:
    with open(file_name) as json_file:
        data = json.load(json_file)
        results.update(data)
print(len(results))

normalize_passage_scores(data_df)

for i in range(11):
    alpha = round(i*.10,2)
    beta = round((10-i)*.10,2)
    print(f'retriever weight: {alpha}, reader weight: {beta}')
    ranked_answer, order_answers = rank_by_confidence(results, data_df, alpha=alpha, beta=beta)
    # orig_scores, rerank_scores = compute_recall(data_df, order_answers)
    data_df = update_data(data_df, order_answers)
    mean = data_df.apply(doc_match_at_k, axis=1).mean()

    print(mean)
    # print(f'orig: {orig_scores}')
    # print(f'rerank: {rerank_scores}')

# data_df.to_csv("/dccstor/srosent3/reranking/mf-coga/experiments/sap_reranking/output/dataset-retrieval-reranking=reader/" + "/output.csv", header=True)