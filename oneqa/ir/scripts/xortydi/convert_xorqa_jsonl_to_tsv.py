#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import json
import pandas as pd

logger=logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


def handle_args():
    usage='Convert XORQA queries jsonl to tsv'	
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--queries_jsonl_file', type=str, required=True, help="path to the xorqa question jsonl") 
    parser.add_argument('--output_file', type = str, required=True, help="path to output file")  

    args=parser.parse_args()
    return args


def yield_json_lines(srcFn: str):
    with open(srcFn) as srcIn:
        for q in srcIn:
            yield json.loads(q)


def handle_queries(q_file: str, outfn: str):
    logger.info(f'read_queries {q_file}')
    dfq=pd.DataFrame.from_records(yield_json_lines(q_file)).reset_index()
    cols=['id','question']
    dfq[cols].to_csv(outfn, sep='\t', header=None, index=None)
    logger.info(f'Wrote {outfn}')


def main():
    args=handle_args()
    handle_queries(args.queries_jsonl_file, args.output_file)

# do main
if __name__=='__main__':
   main()
