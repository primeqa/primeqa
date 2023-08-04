
import gzip
import json
import pandas as pd
import re

# load ir results
# load gold data (in coga csv format?)
# 1. keep examples as negative if ir score is lower than gold passage 
# 2. use rouge to decide negative?

ir_file = "/dccstor/srosent3/reranking/sap_genq/ir_neg_squad/full/synthetic_questions_pos_retrieval.jsonl.gz"
gold_file = "/dccstor/srosent3/reranking/sap_genq/ir_neg_squad/full/synthetic_questions_squadformat_pos.jsonl"

gold_data = []

keep_count = 2

with open(gold_file) as f:
    for line in f:
        json_line = json.loads(line)
        gold_data.append(json_line)

neg_examples = []
less_negs = 0

format = "nq"

with gzip.open(ir_file, 'r') as f:
    for line in gold_data:
        seen_gold = False
        count = 0
        ir_data = json.loads(f.readline())

        for answer in ir_data['answers']:
            if count >= keep_count:
                break
            if answer['id'].startswith(line['document_id']):
                seen_gold = True
            elif seen_gold:
                sections = re.split('\n{5}\n+', answer['text'])

                document_offsets_start = []
                document_offsets_end = []
                document_new = ""
                start = 0
                end = 0

                for section in sections:
                    sentences = re.split('\n+', section)
                    document_offsets_start.append(start)
                    for sentence in sentences:
                        end = len(sentence) + start
                        document_new += sentence + "\n"
                        start = len(document_new)
                    document_offsets_end.append(end)

                output_format = {}
                output_format['example_id'] = line['id'] + "_irneg_" + answer['id']
                output_format['document_id'] = str(answer['id'][:answer['id'].index("-")])
                output_format['question'] = line['question']
                output_format['context'] = [document_new]
                output_format['max_rouge'] = -1.0

                if format == 'squad':
                    output_format['answers'] = {} 
                    output_format['answers']['text'] = [""]
                    output_format['answers']['answer_start'] = [-1]
                else:
                    output_format['target'] = {}
                    output_format['target']['start_positions'] = [-1]
                    output_format['target']['end_positions'] = [-1]
                    output_format['target']['passage_indices'] = [-1]
                    output_format['target']['yes_no_answer'] = ["NONE"]
                    output_format['passage_candidates'] = {}
                    output_format['passage_candidates']['start_positions'] = document_offsets_start
                    output_format['passage_candidates']['end_positions'] = document_offsets_end
                neg_examples.append(output_format)
                count += 1
        if count < keep_count:
            if count == 0:
                less_negs += 1
            else:
                print(f"less negs {count}")

# print(less_negs)
with open("/dccstor/srosent3/reranking/sap_genq/nq_format/train/synthetic_questions_baseformat_train_neg.jsonl", "w", encoding='utf-8') as outfile:
    for line in neg_examples[:-1000]:
        outfile.write(json.dumps(line) + "\n")
with open("/dccstor/srosent3/reranking/sap_genq/nq_format/dev/synthetic_questions_baseformat_dev_neg.jsonl", "w", encoding='utf-8') as outfile:
    for line in neg_examples[-1000:]:
        outfile.write(json.dumps(line) + "\n")