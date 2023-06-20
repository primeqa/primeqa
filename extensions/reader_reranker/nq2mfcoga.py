# given the NQ formatted ranking files, convert to mf-coga output.csv format
# !!! NOTE: don't use, this is still in progress !!!
import pandas as pd 

nq_dir = "/dccstor/colbert-ir/bsiyer/vectordb/indexes/search_dpr_benchmark_1k_q_squad_xor"

info_df = pd.read_json(nq_dir + "/ranked_passages.tsv.annotated.metrics")

qas_df = pd.read_json("/dccstor/colbert-ir/bsiyer/vectordb/data/benchmark_1k_q/qas_with_gold_passages.jsonl", lines=True)
ranking_df = pd.read_csv(info_df['arguments']['ranking'], delimiter="\t", names=['qid','pid','rank','score'])
collection_df = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/data/benchmark_1k_q/psgs.tsv", delimiter="\t")
collection_df_to_id = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/data/benchmark_1k_q/psg_index_to_id.tsv", delimiter="\t")

# query_id,query,gold_document_ids,gold_passages,gold_responses,document_ids,document_titles,
# passages,passage_ids,passage_scores,passage_word_count,passage_word_count_mean,passage_word_count_std,
# passage_word_count_median,n_passages,m_document_ids,order

output = []

for i, question in qas_df.iterrows():
    query_info = {}
    query_info['query_id'] = question['qid']
    query_info['query'] = question['question']
    query_info['gold_document_ids'] = question['gold_passages']
    query_info['gold_passages'] = []
    for gold_passage in question['gold_passages']:
        idx = collection_df_to_id[collection_df_to_id['id'] == question['gold_passages'][0]].iloc[0]['idx']
        document = collection_df[collection_df['id'] == collection_df_to_id[collection_df_to_id['id'] == question['gold_passages'][0]].iloc[0]['idx']]
        query_info['gold_passages'].append(document['text'])
    query_info['gold_responses'] = question['answers']

    # get info for top 40 docs   
    passages = ranking_df[ranking_df['qid'] == question['qid']][ranking_df['rank'] < 40]

    document_ids = []
    document_titles = []
    passage_ids = []
    passage_scores = []
    order = []
    for j, passage in passages.iterrows():
        passage_scores.append(passage['score'])
        passage_ids.append(passage['pid'])
        passage_ids.append(passage['pid'])
  