import sys
import gzip
import json
from tqdm import tqdm
import argparse

def handle_args():
    usage='Convert from MRQA to PrimeQA custom format for QG train/eval'	
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--input_file', type=str, required=True, help="path to MRQA gzip file") 
    parser.add_argument('--output_file', type = str, required=True, help="path to output jsonl file")  

    args=parser.parse_args()
    return args

def main(source_path, dest_path):
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
                        'context': c,
                        'question': q,
                        'answers': a
                    }
                )

    with open(dest_path, 'w') as outfile:
        for e in tqdm(examples, desc="Writing to output file in PrimeQA format", ncols=100):
            json.dump(e, outfile)
            outfile.write('\n')
    print(f"Wrote {dest_path}")
        
# do main
if __name__=='__main__':
    args = handle_args()
    main(args.input_file, args.output_file)
    
