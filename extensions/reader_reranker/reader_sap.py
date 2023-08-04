import pandas as pd
import json
import sys
import ast
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers

default_rank_dir = "/dccstor/srosent3/reranking/mf-coga-new/mf-coga/experiments/output/sap/sap_reranking/task_output/dataset-retrieval-reranking=none" #sys.argv[1]
# "/dccstor/colbert-ir/bsiyer/vectordb/weaviate/benchmark_1k_q_PassageAllMiniLML6v2/" 

data_df = pd.read_csv(default_rank_dir + "/output.csv", header=0)

# load model from model hub
from primeqa.components.reader.extractive import ExtractiveReader

#reader = ExtractiveReader(model="PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110", max_num_answers=40)
#reader = ExtractiveReader(model="/dccstor/srosent2/primeqa/experiments/reader_reranker/2epochRoberta/exclude_passage_answers_discarddups_la_as_sa_moreneg1040_2waylossv2/output/", max_num_answers=40, scorer_type=SupportedSpanScorers.SCORE_DIFF_BASED.value)
reader = ExtractiveReader(model="/dccstor/srosent2/primeqa/experiments/reader_reranker/2epochRoberta/SAPnew/NQftSAP_lr4e-5_1ep-.20-0.20negs/output/", max_num_answers=40, scorer_type=SupportedSpanScorers.SCORE_DIFF_BASED.value, n_best_size=1, max_answer_length=256)
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

def get_passages(data):
    questions = []
    contexts = []
    example_ids = []
    pids = []
    for i, row in data.iterrows():
        passages = ast.literal_eval(row['passages'])
        q_pids = ast.literal_eval(row['document_ids'])
        questions.append(row['query'])
        contexts.append(passages)
        example_ids.append(f'{row["query_id"]}')
        pids.append(q_pids)
    return questions, contexts, example_ids, pids

start = 0 #sys.argv[2]
end = 3 #sys.argv[3]

print(f'start: {start}, end: {end}')
q, c, e, p = get_passages(data_df[int(start):int(end)])

result = run(q,c,e,p)

output_dir = default_rank_dir # sys.argv[4]
with open(f'{output_dir}/reader_answersnewftSAPlr4e-5negsnqformat.json', "w") as outfile: # _{start}-{end}.json
    json.dump(result, outfile)

