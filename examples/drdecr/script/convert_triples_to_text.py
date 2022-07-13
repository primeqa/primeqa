#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import logging
import pandas as pd
import json
import csv
import ujson

from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message

def handle_args():
    usage='usage'
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--triples', required=True)
    parser.add_argument('--queries', required=True)
    parser.add_argument('--collection', required=True)

    parser.add_argument('--out','-o',required=True)  # prefix
    #parser.add_argument('--num_out', '-n', default=50)  # restrict output list of contexts to top n

    args=parser.parse_args()
    return args

def _load_triples(path, rank, nranks):
        """
        NOTE: For distributed sampling, this isn't equivalent to perfectly uniform sampling.
        In particular, each subset is perfectly represented in every batch! However, since we never
        repeat passes over the data, we never repeat any particular triple, and the split across
        nodes is random (since the underlying file is pre-shuffled), there's no concern here.
        """
        print_message("#> Loading triples...")

        triples = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                if line_idx % nranks == rank:
                    qid, pos, neg = ujson.loads(line)
                    triples.append((qid, pos, neg))

        return triples

def _load_queries(path):
        print_message("#> Loading queries...")

        queries = {}

        with open(path) as f:
            for line in f:
                qid, query = line.strip().split('\t')
                qid = int(qid)
                queries[qid] = query

        return queries

def _load_collection(path):
        print_message("#> Loading collection...")

        collection = []

        with open(path) as f:
            for line_idx, line in enumerate(f):
                pid, passage, title, *_ = line.strip().split('\t')
                assert pid == 'id' or int(pid) == line_idx

                passage = title + ' | ' + passage
                collection.append(passage)

        return collection

def main():
    args=handle_args()

    # no sampling (for now)
    rank = 0
    nranks = 1

    triples = _load_triples(args.triples, rank, nranks)
    queries = _load_queries(args.queries)
    collection = _load_collection(args.collection)

    output_list = []

    for position, triple in enumerate(triples):
        query, pos, neg = triple
        query, pos, neg = queries[query], collection[pos], collection[neg]
        output_list.append([query, pos, neg])

    with open(args.out, 'w') as out_file:
        print_message("#> writing " + args.out)
        tsv_writer = csv.writer(out_file, delimiter='\t')
        tsv_writer.writerows(output_list)

# do main
if __name__=='__main__':
   main()