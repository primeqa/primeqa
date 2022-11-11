
# 1. given files from dev, store which ones are in each split in a file. 
# 2. given a file of ids, load into two splits (from Google TyDi)

import glob
import json
import gzip
import logging
from venv import create
import argparse

def get_ids_from_files(file_name, f_out, split):
    
    logging.info(file_name)
    for line in gzip.open(file_name, 'rt', encoding='utf-8'):
        if len(line.strip()) > 0:
            f_out.write(str(json.loads(line)['example_id']) + "," + split + "\n")

# This was used to create the correct split files. No one should need to run this again.
def create_split_files(input_file, output_dir):
    split1 = glob.glob(input_file + "*0[0-4]*.gz")
    split2 = glob.glob(input_file + "*0[5-9]*.gz")

    with open(output_dir + "/tydiqa-dev-split.txt",'w') as f_out:
        for f_in in split1:
            get_ids_from_files(f_in,f_out, "split1")
        for f_in in split2:
            get_ids_from_files(f_in, f_out, "split2")

def generate_splits(input_file, split_file, output_dir):
    # load the splits. for each example, check which split it belongs in and store in appropriate file
    example_ids = {}
    with open(split_file + "/tydiqa-dev-split.txt", 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(",")
            example_ids[data[0]] = data[1][:-1]

    split_files = {} 
    split_files["split1"] = gzip.open(output_dir + "/tydiqa-v1.0-dev-00.jsonl.gz", 'wt', encoding='utf-8')
    split_files["split2"] = gzip.open(output_dir + "/tydiqa-v1.0-dev-01.jsonl.gz", 'wt', encoding='utf-8')

    for line in gzip.open(input_file, 'rt', encoding='utf-8'):
        if len(line.strip()) > 0:
            data = json.loads(line)
            split_files[example_ids[str(data["example_id"])]].write(line)
    split_files["split1"].close()
    split_files["split2"].close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='create the data split for training the tydi score normalizer')
    parser.add_argument('--output_dir',
        help='the output directory',
        type=str)  
    parser.add_argument('--original_tydi_dir',
        help='directory the original tydi is located in',
        type=str)  
    args = parser.parse_args()
    return args
    

def main():
    args = parse_arguments()

    split_dir = "examples/boolqa"
    # set this to the location of the original tydi downloaded from google:
    original = args.original_tydi_dir + "/tydiqa-v1.0-dev.jsonl.gz"
    # set this to the location to save the new split
    output_dir = args.output_dir
    
    generate_splits(original, split_dir, output_dir)

if __name__ == '__main__':
    main()