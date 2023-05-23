#!/bin/env python
# -*- coding: utf-8 -*-

import csv
import argparse
import json
import os
import re
import random
import jsonlines

def handle_args():
    usage='usage'
    parser=argparse.ArgumentParser(usage)
    # parser.add_argument('--ground_truth_fn', '-g', required=True)
    parser.add_argument('--rankings_fn', '-r', required=True)
    parser.add_argument('--questions_fn', '-q', required=True)
    parser.add_argument('--passages_fn', '-p', required=True)


    parser.add_argument('--passid_map_fn', '-m', required=True)
    parser.add_argument('--output_fn','-o', required=True)

    parser.add_argument('--num_negatives_per_query', '-N', type=int, default=10, )
    parser.add_argument('--num_queries', '-Q', type=int, default=1000, )

    args=parser.parse_args()
    return args

def main():
    random.seed(12345)
    args=handle_args()

    rankings = {}
    with open(args.rankings_fn) as in_f:
        reader = csv.reader(in_f, delimiter='\t')
        for pos, row in enumerate(reader):
            if row[0] not in rankings:
                rankings[row[0]] = {}
                rankings[row[0]]['pos'] = []
                rankings[row[0]]['neg'] = []
            if row[3] == '1':
                rankings[row[0]]['pos'].append(row[1])
            else:
                rankings[row[0]]['neg'].append(row[1])

    questions = []
    with open(args.questions_fn) as in_f:
        reader = csv.reader(in_f, delimiter='\t')
        for pos, row in enumerate(reader):
            questions.append(row)

    passages = []
    with open(args.passages_fn) as in_f:
        reader = csv.reader(in_f, delimiter='\t')
        for pos, row in enumerate(reader):
            passages.append(row)

    passid_map = []
    pass_recs = []
    seen_pass = set()

    num_queries_processed = 0
    num_pass_out = 0

    def add_records(pass_num, num_pass_out):
        if not pass_num in seen_pass and int(pass_num) < len(passages):
            seen_pass.add(pass_num)
            doc = passages[int(pass_num)]
            num_pass_out += 1
            rec = [num_pass_out, doc[1], doc[2]]
            pass_recs.append(rec)
            passid_map.append([num_pass_out, pass_num])
        return num_pass_out

    for qnum in rankings.keys():
        if len(rankings[qnum]['pos']) > 0:
            pass_num = rankings[qnum]['pos'][0]
            num_pass_out = add_records(pass_num, num_pass_out)

            for npos in range(args.num_negatives_per_query):
                pass_num = rankings[qnum]['neg'][npos]
                num_pass_out = add_records(pass_num, num_pass_out)

            '''if not pass_num in seen_pass and int(pass_num) < len(passages):
                seen_pass.add(pass_num)
                doc = passages[int(pass_num)]
                rec = [num_pass_out, doc[1], doc[2]]
                pass_recs.append(rec)
                passid_map.append([num_pass_out, pass_num])
                num_pass_out += 1

            for npos in range(args.num_negatives_per_query):
                pass_num = rankings[qnum]['neg'][npos]
                if not pass_num in seen_pass and int(pass_num) < len(passages):
                    seen_pass.add(pass_num)
                    doc = passages[int(pass_num)]
                    rec = [num_pass_out, doc[1], doc[2]]
                    pass_recs.append(rec)
                    passid_map.append([num_pass_out, pass_num])
                    num_pass_out += 1
            '''
            num_queries_processed += 1
            if num_queries_processed >= args.num_queries:
                break

    with open(args.output_fn, 'w') as out_f:
        tsv_writer = csv.writer(out_f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
        tsv_writer.writerow(['id', 'text', 'title'])
        tsv_writer.writerows(pass_recs)


    with open(args.passid_map_fn, 'w') as out_f:
        tsv_writer = csv.writer(out_f, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
        tsv_writer.writerows(passid_map)


    print(f'Written {len(pass_recs)} passage records')


# do main
if __name__=='__main__':
   main()
