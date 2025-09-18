import pandas as pd
import glob
import json
import tqdm 

def read_jsonl(filename: str, encoding="utf-8"):
    with open(filename, mode="r", encoding=encoding) as fp:
        content = [json.loads(line.rstrip("\n").strip()) for line in fp]

    return content

def write_jsonl(filename: str, content: list, encoding="utf-8"):
    with open(filename, mode="w", encoding=encoding) as fp:
        for example in content:
            json.dump(example, fp)
            fp.write("\n")

passages = pd.read_csv("/dccstor/srosent2/primeqa/data/corpora/nq_passages_unique.tsv", delimiter="\t", header=0)
question_files = glob.glob("/proj/srosent3/reinforcement_learning/rag_rlvr/nq_mr2/answerable_adjusted_passages/nq*.jsonl")

print(len(question_files))

for question_file in question_files:
    questions = read_jsonl(question_file)
    print(question_file)
    print(f"num questions start: {len(questions)}")
    missing_passages = 0
    missing_target = 0
    questions_final = []
    for question in tqdm.tqdm(questions, total=len(questions)):
        keep = True
        for context in question["contexts"]:
            match = passages[passages['id'] == context['document_id']]
            if len(match) == 0:
                keep = False
                missing_passages += 1
                # print(f"missing passages! {context['document_id']}")
                continue
            match = match.iloc[0]
            has_target = False
            for target in question['targets']:
                if target['text'] in match['text']:
                    has_target = True
            if not has_target:
                missing_target += 1
                # print(f"missing target! {question['task_id']}: {question['targets']} {match['text']}")
                keep = False
                continue
        if keep:
            questions_final.append(question)
    print(f"missing passages: {missing_passages}, missing target: {missing_target}")
    print(f"num questions end: {len(questions_final)}")
    write_jsonl(question_file.replace("answerable_adjusted_passages","answerable_final"),questions_final)