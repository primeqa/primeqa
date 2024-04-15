from tqdm import tqdm
import glob 
import json
import pandas as pd
import ast

# Step 1: Read basic multi-turn
# Step 2: Add longNQ doc ids
# Step 3: Output in multi-turn conversation format

def read_jsonl(filename: str, encoding="utf-8"):
    with open(filename, mode="r", encoding=encoding) as fp:
        content = [json.loads(line.rstrip("\n").strip()) for line in fp]

    return content

def write_jsonl(filename: str, content: dict, encoding="utf-8"):
    with open(filename, mode="w", encoding=encoding) as fp:
        for example in content:
            json.dump(content[example], fp, indent=4)

split = "dev"

output_dir = f"/dccstor/srosent1/human_ai_eval/human_eval_data/processing/en/longNQ/{split}"

multi_turn_conversations = read_jsonl(f"{output_dir}/multi_turn_gold_processed.jsonl")

retrieval_passages = pd.read_csv("/dccstor/srosent2/generative/appen/final/longNQ/passages_for_index/LongNQ_train_dev_test_passages_w_unanswerable_ids.tsv", delimiter='\t', names=['doc_id','text', 'title', 'question_id','split'], header=0)

questionid2docid = {}

for i, row in retrieval_passages.iterrows():
    for qid in ast.literal_eval(row['question_id']):
        if qid not in questionid2docid:
            questionid2docid[qid] = []
        questionid2docid[qid].append(row['doc_id'])

conversations = {}
contexts = {}

for mt_conversation in tqdm(multi_turn_conversations):
    
    if mt_conversation['conversation_id'] not in conversations:
        conversation = {}
        conversation['conversation_id'] = mt_conversation['conversation_id']
        conversation['collection'] = "Wikipedia"
        conversation['dataset'] = 'LongNQ'
        conversation['conversation'] = []
    else:
        conversation = conversations[mt_conversation['conversation_id']]

    conversation['conversation'].append({
        "speaker": "user",
        "metadata": {
            "author_type": "human",
            "author_id": "NQ",
            "question_id": mt_conversation["task_id"]
        },
        "text": mt_conversation['input'][-1]['text']
    })

    conversation['conversation'].append({
        "speaker": "agent",
        "metadata": {
            "author_type": "human",
            "author_id": "longNQ",
            "question_id": mt_conversation["task_id"],
            "answerability": mt_conversation['targets'][-1]['enrichments']['answerability']
        },
        "text": mt_conversation['targets'][-1]['text'] if mt_conversation['targets'][-1]['enrichments']['answerability'] == "ANSWERABLE" else "I don't know the answer.",
        "evidence_context": questionid2docid[mt_conversation['task_id'][:-3]] if mt_conversation['task_id'][:-3] in questionid2docid else [],
        "retrieved_context": []
    })
    conversations[mt_conversation['conversation_id']] = conversation

write_jsonl(f"{output_dir}/multi_turn_gold_conversation.jsonl", conversations)