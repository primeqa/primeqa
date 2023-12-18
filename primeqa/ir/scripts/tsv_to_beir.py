# convert tsv format to beir format
import pandas as pd
import glob

# load tsv questions and passages
# output queries, corpus, qrels beir format

passages_file = "/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/passages.tsv"

passages_df = pd.read_csv(passages_file, delimiter='\t', names=["_id", "text", "title"], header=0, dtype=str)
passages_df['metadata'] = {}
passages_df.to_json("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/beir_format/corpus.jsonl", lines=True, orient="records")

questions_file = glob.glob("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/*/*answerable.tsv")

# queries
# _id "text" "metadata-answer"

queries = []

for question_file in questions_file:
    qrels = []

    questions = pd.read_csv(question_file, delimiter="\t", names=["_id","text","doc-id-list","answers"], header=0, dtype=str)

    for i, row in questions.iterrows():
        if not pd.isna(row['answers']):
            queries.append({'_id': str(row['_id']), 'text': row['text'], 'metadata': {'answers':row['answers']}})
        else:
            queries.append({'_id': str(row['_id']), 'text': row['text'], 'metadata': {}})
        if not pd.isna(row['doc-id-list']):
            corpusids = passages_df[passages_df['_id'] == row['doc-id-list']]['_id'].tolist()

            for corpusid in corpusids:
                # add qrel
                qrels.append([str(row["_id"]), str(corpusid), 1])

    # qrels
    # query-id        corpus-id       score
    qrels_df = pd.DataFrame(qrels, columns=["query-id","corpus-id","score"]).to_csv(f"/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/beir_format/qrels/{question_file[question_file.rindex('/'):-4]}.tsv",sep='\t', index=False)
pd.DataFrame(queries).to_json("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/beir_format/queries.jsonl", lines=True, orient='records')