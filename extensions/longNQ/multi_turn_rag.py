import watson_nlp
from watson_nlp.blocks.syntax import izumo
from watson_nlp.blocks.entity_mentions import BERT
# from rouge_score import rouge_scorer
# import pandas as pd
# import random
from tqdm import tqdm
import glob 
import json

# Step 1: Get count of all mentions and nouns from split.
# Step 2: Piece together questions with matching mentions/nouns as multiturn setup 
# Step 3: Output in model runner format

def read_jsonl(filename: str, encoding="utf-8"):
    with open(filename, mode="r", encoding=encoding) as fp:
        content = [json.loads(line.rstrip("\n").strip()) for line in fp]

    return content

def write_jsonl(filename: str, content: list, encoding="utf-8"):
    with open(filename, mode="w", encoding=encoding) as fp:
        for example in content:
            fp.write(json.dumps(example) + "\n")

def get_mentions_and_np(all_mentions_np, text, example_id):
    syntax_analysis_text_en = syntax_model_en.run(text, parsers=('token',))
    mentions_text_prediction = entity_model.run(syntax_analysis_text_en)

    # get mentions
    for mention in mentions_text_prediction.mentions:
        if mention.type == 'Date':
            continue
        if mention.text.lower() not in all_mentions_np:
            all_mentions_np[mention.text.lower()] = []
        all_mentions_np[mention.text.lower()].append(example_id)
    # get nouns
    np_predictions = np_model.run(text)

    for np_prediction in np_predictions.noun_phrases:
        if np_prediction.text.lower() not in all_mentions_np:
            all_mentions_np[np_prediction.text.lower()] = []
        all_mentions_np[np_prediction.text.lower()].append(example_id)
    return all_mentions_np

# Load the syntax model for English
model_path = watson_nlp.download('noun-phrases_rbr_en_stock', parent_dir='/dccstor/srosent2/watson_nlp') #syntax_izumo_en_stock')
np_model = watson_nlp.load(model_path)

model_path = watson_nlp.download('entity-mentions_bert_multi_stock', parent_dir='/dccstor/srosent2/watson_nlp') #'entity-mentions_rbr_en_stock') #syntax_izumo_en_stock')
# model_path = watson_nlp.download('entity-mentions_transformer_multilingual_slate.270m')
entity_model = watson_nlp.load(model_path)

syntax_model_en = watson_nlp.load(watson_nlp.download('syntax_izumo_en_stock', parent_dir='/dccstor/srosent2/watson_nlp'))

# rouge = rouge_scorer.RougeScorer(rouge_types=['rougeLsum'], split_summaries=True)

split = "dev"
longnq_files = glob.glob(f"/dccstor/srosent1/human_ai_eval/human_eval_data/processing/en/longNQ/{split}/processed.jsonl")
longNQdata = []
for longnq_file in longnq_files:
    longNQdata.extend(read_jsonl(longnq_file))

all_target_mentions_np = {}
all_question_mentions_np = {}
questions_by_id = {}


with tqdm(total=len(longNQdata)) as pbar:    
    for example in longNQdata:
        pbar.update(1)
        questions_by_id[example['task_id']] = example
        
        question = example['input'][0]['text']
        target = example['targets'][-1]['text']
        if len(example['targets']) > 2:
            print(len(example['targets']))
        all_target_mentions_np = get_mentions_and_np(all_target_mentions_np, target, example['task_id'])
        all_question_mentions_np = get_mentions_and_np(all_question_mentions_np, question, example['task_id'])

print(len(all_target_mentions_np))
print(len(all_question_mentions_np))

grouped_questions = {}

for question_mentions_np in all_question_mentions_np:
    # only occurs once
    if len(all_question_mentions_np[question_mentions_np]) == 1 and question_mentions_np not in all_target_mentions_np:
        continue
    # too common
    if len(all_question_mentions_np[question_mentions_np]) > 4:
        continue

    if question_mentions_np not in grouped_questions:
        grouped_questions[question_mentions_np] = set()

    for qid in all_question_mentions_np[question_mentions_np]:
        grouped_questions[question_mentions_np].add(qid)
    if question_mentions_np in all_target_mentions_np and len(grouped_questions[question_mentions_np]) < 4:
        for qid in all_target_mentions_np[question_mentions_np]:
            grouped_questions[question_mentions_np].add(qid)
            if len(grouped_questions[question_mentions_np]) >= 4:
                break

print(len(grouped_questions))

multi_turn_all = []
seen = set()

index = 0

for question_group in grouped_questions:
    if len(grouped_questions[question_group]) <= 1:
        continue
    multi_turn_examples = []
    for qid in grouped_questions[question_group]:
        if qid in seen:
            continue
        seen.add(qid)
        example = questions_by_id[qid].copy()
        example['task_id'] += f"::{len(multi_turn_examples)}"
        example['conversation_id'] = f"LongNQ_MT_{index}::{question_group}"
        example['turn'] = str(len(multi_turn_examples))
        if len(multi_turn_examples) == 0:
            multi_turn_examples.append(example)
        else:
            last_turn = multi_turn_examples[-1]
            new_input = last_turn['input'].copy()
            last_target = last_turn['targets'][-1].copy()
            last_target['speaker'] = 'target'
            new_input.append(last_target)
            new_input.extend(example['input'])
            example['input'] = new_input
            multi_turn_examples.append(example)
    if len(multi_turn_examples) > 1:
        multi_turn_all.extend(multi_turn_examples)
    index += 1

print(len(multi_turn_all))

write_jsonl(f"/dccstor/srosent1/human_ai_eval/human_eval_data/processing/en/longNQ/{split}/multi_turn_gold_processed.jsonl", multi_turn_all)