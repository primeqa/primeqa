import json
import os
import argparse
from tqdm import tqdm
import csv
import logging
import gzip
logger = logging.getLogger(__name__)


def handle_args():
    parser = argparse.ArgumentParser(description='Convert XOR corpus of 100 token passages into JSONL that can be processed by Pyserini'
)
    parser.add_argument('--input_file', required=True, help='Path to the corpus tsv gzip file')
    parser.add_argument('--output_dir', required=True, help='Path to the output directory')
    args=parser.parse_args()
    return args


def main():
    args = handle_args()
    in_file = args.input_file 
    out_dir= args.output_dir  
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_file = os.path.join(out_dir, "psgs_w100_pyserini.jsonl")
    logger.info(f'Reading {in_file}')
    with gzip.open(in_file, 'rt') as f, open(out_file, 'w') as outp:
        reader = csv.DictReader(f, delimiter='\t')
        for row in tqdm(reader):
            json_string = json.dumps({
                'id': row['id'],
                'contents': f'{row["title"]}\t{row["text"]}'
            })
            outp.write(f'{json_string}\n')
    logger.info(f'Wrote {out_file}')


if __name__ == '__main__':
    main()
    print("Success...")


