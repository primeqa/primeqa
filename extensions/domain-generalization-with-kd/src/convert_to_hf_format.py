import sys
import gzip
import json
from tqdm import tqdm

source_path = sys.argv[1]
dest_path = sys.argv[2]

examples = []
with gzip.open(source_path, 'r') as infile:
    is_header = True
    for line in tqdm(infile, desc="Reading input file", ncols=100):
        if is_header:
            is_header = False
            continue
        e = json.loads(line)
        c = e['context']
        for qa in e['qas']:
            id_ = qa['id'] if 'id' in qa else qa['qid']
            q = qa['question']
            a = {
                'text': [],
                'answer_start': []
            }
            for item in qa['detected_answers']:
                a['text'].append(item['text'])
                a['answer_start'].append(item['char_spans'][0][0])
            examples.append(
                {
                    'id': id_,
                    'title': None,
                    'context': c,
                    'question': q,
                    'answers': a
                }
            )

with open(dest_path, 'w') as outfile:
    for e in tqdm(examples, desc="Writing to output file in HF format", ncols=100):
        json.dump(e, outfile)
        outfile.write('\n')
