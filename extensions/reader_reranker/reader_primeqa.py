import pandas as pd
import json
import sys
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers

default_rank_dir = sys.argv[1]
# "/dccstor/colbert-ir/bsiyer/vectordb/weaviate/benchmark_1k_q_PassageAllMiniLML6v2/" 

info_df = pd.read_json(default_rank_dir + "/ranked_passages.tsv.annotated.metrics")

qas_df = pd.read_json(info_df['arguments']['qas'], lines=True)
ranking_df = pd.read_csv(info_df['arguments']['ranking'], delimiter="\t", names=['qid','pid','rank','score'])
collection_df = pd.read_csv(info_df['arguments']['collection'], delimiter="\t")

# load model from model hub
from primeqa.components.reader.extractive import ExtractiveReader

reader = ExtractiveReader(model="PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110", max_num_answers=100, scorer_type=SupportedSpanScorers.SCORE_DIFF_BASED.value)
reader.load()


# list of questions and list of list of contexts (list for each question)
def run(questions, contexts, example_ids, passage_ids):
    reader_answers = reader.predict(
        questions, contexts, example_ids=example_ids
    )
    result = {}
    index = 0
    for i, answers_i in reader_answers.items():
        i_result = {}
        for answer in answers_i:
            answer['passage_index'] = passage_ids[index][answer['passage_index']] #[i.index('.')+1:])]
        i_result["answers"] = answers_i
        i_result["passages"] = contexts[index] #[i.index('.')+1:])]
        result[i] = i_result
        index+= 1
    return result

def get_passages(qas, collection, ranking, num_passages=100):
    questions = []
    contexts = []
    example_ids = []
    pids = []
    for i, row in qas.iterrows():
        q_passages = ranking[ranking['qid'] == row['qid']][:num_passages]
        
        passages = []
        q_pids = []
        count = 0
        for j, passage in q_passages.iterrows():
            # questions.append(row['question'])
            passages.append(collection.iloc[int(passage['pid'])-1]['title'] + "\n" + collection.iloc[int(passage['pid'])-1]['text'])
            q_pids.append(str(int(passage['pid'])))
            # example_ids.append(f'{row["qid"]}.{count}')
            count+= 1
        questions.append(row['question'])
        contexts.append(passages)
        example_ids.append(f'{row["qid"]}')
        pids.append(q_pids)
    return questions, contexts, example_ids, pids

start = sys.argv[2]
end = sys.argv[3]

print(f'start: {start}, end: {end}')
q, c, e, p = get_passages(qas_df[int(start):int(end)], collection_df, ranking_df)

result = run(q,c,e,p)

output_dir = sys.argv[4]
with open(f'/dccstor/srosent3/reranking/{output_dir}/reader_answers_diffbased_{start}-{end}.json', "w") as outfile:
    json.dump(result, outfile)

