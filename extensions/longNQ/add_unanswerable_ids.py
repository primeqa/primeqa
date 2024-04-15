import pandas as pd
import glob
import json
import ast
import tqdm

def read_jsonl(filename: str, encoding="utf-8"):
    with open(filename, mode="r", encoding=encoding) as fp:
        content = [json.loads(line.rstrip("\n").strip()) for line in fp]

    return content

retrieval_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/LongNQ_train_dev_test_passages_wids.tsv", delimiter='\t', names=['doc_id','text', 'title', 'question_id','split'], header=0)

longnq_files = glob.glob(f"/dccstor/srosent2/generative/appen/final/longNQ/*/longNQ_*_unanswerable.jsonl")



for longnq_file in longnq_files:
    missing = 0
    print(longnq_file)
    longnq_data = read_jsonl(longnq_file)

    for example in tqdm.tqdm(longnq_data):
        passage = retrieval_passages[retrieval_passages['text'] == example['passages'][0]['text']]

        if len(passage) == 1:
            question_ids = ast.literal_eval(passage.iloc[0]['question_id'])
            question_ids.append(example['id'])
            retrieval_passages.loc[passage.index[0]]['question_id'] = str(question_ids)
        else:
            missing+= 1
            # print(f"missing gold passage for this unanswerable question {longnq_file}")
            # passages = retrieval_passages[retrieval_passages['title'] == example['passages'][0]['title']]
            # print(example)

    print(f"missing: {missing}")
retrieval_passages.to_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/LongNQ_train_dev_test_passages_w_unanswerable_ids.tsv", sep='\t')
