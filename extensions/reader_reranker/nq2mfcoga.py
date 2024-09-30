# given the NQ formatted ranking files, convert to mf-coga output.csv format
import pandas as pd 

qas_df = pd.read_json("/dccstor/colbert-ir/bsiyer/vectordb/data/rag_corpus_nq_910/qas.jsonl", lines=True)
# ranking_df = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/indexes/search_dpr_nq910_squad_xor/ranked_passages.tsv", delimiter="\t", names=['qid','pid','rank','score'])
ranking_df = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/indexes/colbert_nq910_xor-squad_roberta_model/ranked_passages.tsv", delimiter="\t", names=['qid','pid','rank','score'])
collection_df = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/data/rag_corpus_nq_910/psgs.tsv", delimiter="\t")

qas_df = pd.read_json("/dccstor/colbert-ir/bsiyer/vectordb/data/benchmark_nq_dev_4k/qas.jsonl", lines=True)
ranking_df = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/indexes/colbert_nq910_xor-squad_roberta_model/ranked_passages.tsv", delimiter="\t", names=['qid','pid','rank','score'])
collection_df = pd.read_csv("/dccstor/colbert-ir/bsiyer/vectordb/data/benchmark_nq_dev_4k/psgs.tsv", delimiter="\t")

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
        # idx = collection_df_to_id[collection_df_to_id['id'] == question['gold_passages'][0]].iloc[0]['idx']
        document = collection_df[collection_df['id'] == int(gold_passage)]
        query_info['gold_passages'].append(document['text'].values[0])
    query_info['gold_responses'] = question['answers']

    # get info for top 40 docs   
    passages = ranking_df[ranking_df['qid'] == question['qid']][ranking_df['rank'] < 40]

    document_ids = []
    document_titles = []
    passage_ids = []
    passage_scores = []
    passage_texts = []
    order = []
    passage_word_count = []
    
    for j, passage in passages.iterrows():
        passage_scores.append(passage['score'])
        passage_ids.append(str(int(passage['pid'])))
        document = collection_df[collection_df['id'] == int(passage['pid'])]
        document_titles.append(document['title'].values[0])
        passage_texts.append(document['text'].values[0])
        passage_word_count.append(len(document['text'].values[0].split(" ")))
        doc_id = str(int(passage['pid']))
        document_ids.append(doc_id)
        order.append(int(passage['rank']))
    query_info['document_ids'] = document_ids
    query_info['document_titles'] = document_titles
    query_info['passages'] = passage_texts
    query_info['passage_ids'] = passage_ids
    query_info['passage_scores'] = passage_scores
    query_info['passage_word_count'] = passage_word_count
    query_info['order'] = order
    output.append(query_info)

df = pd.DataFrame.from_dict(output)

df.to_csv("/dccstor/srosent3/reranking/mf-coga/experiments/sap_reranking_colbert/output/dataset-retrieval/output.csv")
#df.to_csv("/dccstor/srosent3/reranking/nq-perplexity/output.csv")