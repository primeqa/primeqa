# read in the file and then check if can
# find sentence with high rouge to consider as answer passage in the document.
# Finally, convert to SQuAD format.

import pandas as pd
from primeqa.mrc.metrics.rouge.rouge import ROUGE
import re
import json
import glob

seen_pos = set()
data_pos = []
data_neg = []
questions = []

# # load existing questions with docs ids
# with open('/dccstor/srosent3/reranking/sap_genq/unique_ids/full/synthetic_questions_squadformat.jsonl') as f:
#     for line in f:
#         json_line = json.loads(line)
#         data_pos.add(json_line['document_id'] + json_line['question'])

# # load existing questions with docs ids
# with open('/dccstor/srosent3/reranking/sap_genq/unique_ids/full/synthetic_questions_squadformat_negative.jsonl') as f:
#     for line in f:
#         json_line = json.loads(line)
#         data_neg.add(json_line['document_id'] + json_line['question'])

# with open("/dccstor/srosent3/reranking/sap_genq/unique_ids/full/synthetic_questions_squadformat_negative.jsonl", "w", encoding='utf-8') as outfile:
#     for line in data:
#         outfile.write(json.dumps(line) + "\n")

rouge = ROUGE()

# input_file = "/dccstor/gma/arafat/sap/data/SAP_20230710.ensemble.grounded-and-useful.jsonl"
# input_file = "/dccstor/gma/arafat/sap/data/SAP_20230710.ensemble.not-useful.jsonl"
# input_file = "/dccstor/gma/arafat/sap/data/SAP_20230710.ensemble.not-grounded.jsonl"

files = glob.glob("/dccstor/gma/arafat/sap/data/*.jsonl")

for input_file in files:
    sap_data = pd.read_json(input_file, lines=True)

    count = 0

    for index, row in sap_data.iterrows():
        output_format = {}
        document = re.sub('\n+', '\n', row['document']).split("\n")
        document_new = ""

        document_offsets_start = []
        document_offsets_end = []

        max_rouge = -1.0
        max_sentence = -1.0

        sentence_index = 0
        start = 0
        end = 0
        for sentence in document:
            end = len(sentence) + start
            document_offsets_start.append(start)
            document_offsets_end.append(end)
            document_new += sentence + "\n"
            start = len(document_new)

            if 'grounded-and-useful' in input_file:
                hf_metric, _ = rouge._rougel_score(row['response'], sentence)
                if hf_metric > max_rouge:
                    max_rouge = hf_metric
                    max_sentence = sentence_index
            sentence_index += 1

        # squad format:
        # {"id": "1", "question": "what language is spoken in afghanistan", 
        # "context": "There are between 40 and 59 languages spoken in Afghanistan franca.", 
        # "answers": {"text": ["Dari and Pashto"], "answer_start": [61]}}

        output_format['id'] = input_file[input_file.rindex("ensemble.")+9:-6] + "_" + str(index)
        output_format['document_id'] = row['document_id']
        output_format['question'] = row['query']
        output_format['context'] = document_new
        output_format['answers'] = {} 
        if max_sentence == -1:
            output_format['answers']['text'] = [""]
            output_format['answers']['answer_start'] = [-1]
        else:
            output_format['answers']['text'] = [document_new[document_offsets_start[max_sentence]:document_offsets_end[max_sentence]]]
            output_format['answers']['answer_start'] = [document_offsets_start[max_sentence]]

        if 'grounded-and-useful' in input_file:
            data_pos.append(output_format)
            seen_pos.add(row['document_id'] + row['query'])
        elif row['document_id'] + row['query'] in seen_pos:
            continue
        else:
            data_neg.append(output_format)
            questions.append(row['query'])
        count += 1

        if count % 100 == 0:
            print(count)



# shuffle questions with passages and set answer to empty with answer_start=-1
import random
random.shuffle(questions)

for line in data_neg:
    question = questions.pop()

    if line['question'] == question:
        question = questions.pop()
        questions.append(line['question'])
    line['question'] = question
    # line['answers']['text'] = [""]
    # line['answers']['answer_start'] = [-1]

with open("/dccstor/srosent3/reranking/sap_genq/more_neg/synthetic_questions_squadformat_pos.jsonl", "w", encoding='utf-8') as outfile:
    for line in data_pos:
        outfile.write(json.dumps(line) + "\n")

with open("/dccstor/srosent3/reranking/sap_genq/more_neg/synthetic_questions_squadformat_neg.jsonl", "w", encoding='utf-8') as outfile:
    for line in data_neg:
        outfile.write(json.dumps(line) + "\n")