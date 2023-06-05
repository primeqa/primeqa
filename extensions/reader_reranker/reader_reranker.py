import pandas as pd
import json
import sys

default_rank_dir = sys.argv[1] #"/dccstor/colbert-ir/bsiyer/vectordb/weaviate/benchmark_1k_q_PassageAllMiniLML6v2/"

info_df = pd.read_json(default_rank_dir + "/ranked_passages.tsv.annotated.metrics")

qas_df = pd.read_json(info_df['arguments']['qas'], lines=True)
ranking_df = pd.read_csv(info_df['arguments']['ranking'], delimiter="\t", names=['qid','pid','rank','score'])
collection_df = pd.read_csv(info_df['arguments']['collection'], delimiter="\t")

# load model from model hub
from primeqa.components.reader.extractive import ExtractiveReader

reader = ExtractiveReader(model="PrimeQA/tydiqa-primary-task-xlm-roberta-large", max_num_answers=10)
reader.load()


# list of questions and list of list of contexts (list for each question)
def run(questions, contexts, example_ids, passage_ids):
    reader_answers = reader.predict(
        questions, contexts, example_ids=example_ids
    )
    result = {}
    for i, answers_i in reader_answers.items():
        i_result = {}
        for answer in answers_i:
            answer['passage_index'] = passage_ids[int(i[i.index('.')+1:])]
        i_result["answers"] = answers_i
        i_result["passages"] = contexts[int(i[i.index('.')+1:])]
        result[i] = i_result
    return result

def get_passages(qas, collection, ranking, num_passages=100):
    questions = []
    contexts = []
    example_ids = []
    pids = []
    for i, row in qas.iterrows():
        q_passages = ranking[ranking['qid'] == row['qid']][:num_passages]
        
        passages = []
        count = 0
        for j, passage in q_passages.iterrows():
            questions.append(row['question'])
            contexts.append([collection.iloc[int(passage['pid'])-1]['title'] + "\n" + collection.iloc[int(passage['pid'])-1]['text']])
            pids.append(str(int(passage['pid'])))
            example_ids.append(f'{row["qid"]}.{count}')
            count+= 1
        # questions.append(row['question'])
        # contexts.append(passages)
    return questions, contexts, example_ids, pids

start = sys.argv[2]
end = sys.argv[3]

print(f'start: {start}, end: {end}')
q, c, e, p = get_passages(qas_df[int(start):int(end)], collection_df, ranking_df)

result = run(q,c,e,p)

with open(f'/dccstor/srosent3/reranking/{sys.argv[4]}/reader_answers_{start}-{end}.json', "w") as outfile:
    json.dump(result, outfile)

