# This script converts MRQA 2019-style datasets (https://github.com/mrqa/MRQA-Shared-Task-2019) to the Hugging Face Datasets SQuAD format:
# ```
# python utils/convert_to_hf_format.py <path-to-input-mrqa-style-file> <path-to-output-hf-style-file>
# ```
# The first argument is the path to the input input file (in MRQA format) and the second argument is the output file path which will contain the data in HF Datasets SQuAD-style format.
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

def main():
    with open(dest_path, 'w') as outfile:
        for e in tqdm(examples, desc="Writing to output file in HF format", ncols=100):
            json.dump(e, outfile)
            outfile.write('\n')

if __name__ == '__main__':
    main()
    
