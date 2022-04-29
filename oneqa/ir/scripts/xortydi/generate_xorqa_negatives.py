#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import json
import jsonlines
import numpy.random
from tqdm import tqdm
from oneqa.ir.sparse.retriever import PyseriniRetriever

numpy.random.seed(1234)
logger=logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

from utils import HasAnswerChecker

def handle_args():
    usage='usage'
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--input_file', '-i', required=False, default='/dccstor/jsmc-nmt-01/qa/xor-tydi/train-multilingual-2021-02-25/1/xorqa_dpr_data_query-L_hard_negative-1/dpr_train_data.json')
    parser.add_argument('--output_file', '-o', required=False, default='/dccstor/bsiyer6/OneQA/xortydi_bm25/xortydi_train_data_ir_negs_poss.json')
    parser.add_argument('--question_translations_dir', '-d', required=False,default='/dccstor/colbert-ir/bsiyer/data/xorqa/trans_data_all_langs')
    parser.add_argument('--index_path', '-u', required=False,default='/dccstor/bsiyer6/OneQA/psgs_w100_index')
    parser.add_argument('--max_num_ir_based_negatives', '-n', type=int, default=300)
    parser.add_argument('--max_num_ir_based_positives', '-p', type=int, default=10)
    parser.add_argument('--do_not_run_ir', '-R', action="store_true", default=False)
    parser.add_argument('--do_not_run_match_in_title', '-T', action="store_true", default=False)

    parser.add_argument('--question_records_file', '-q', default=None)
    parser.add_argument('--languages_list', '-l', nargs='+', default=[])

    args=parser.parse_args()
    return args

def init_question_translations(question_translations_dir):
    question_translations = {}
    # this is very XOR-TyDi QA specific
    langs = set(['ar', 'bn', 'fi', 'ja', 'ko', 'ru', 'te'])

    for lang in langs:
        with open(question_translations_dir + '/' + lang + '-en/en.txt') as f:
            en_content = f.readlines()
        with open(question_translations_dir + '/' + lang + '-en/' + lang + '.txt') as f:
            ne_content = f.readlines()

        assert(len(en_content) == len(ne_content))
        for ne, en in zip(ne_content, en_content):
            question_translations[ne.strip()] = en.strip()

    return question_translations


def run_query(query, searcher, max_retrieved):

    hits = searcher.retrieve(query, max_retrieved)
    retrieved = []
    for hit in hits:
        retrieved.append( (hit['doc_id'],hit['score'], hit['title'], hit['text']) )
    return retrieved

def read_jsonlines(file_name):
    lines = []
    print("loading examples from {0}".format(file_name))
    with jsonlines.open(file_name) as reader:
        for obj in reader:
            lines.append(obj)
    return lines


def main():
    args=handle_args()

    answer_checker = HasAnswerChecker()

    searcher = PyseriniRetriever(args.index_path)

    question_translations = init_question_translations(args.question_translations_dir)

    qas=json.load(open(args.input_file))

    if args.question_records_file is not None:
        q_recs=read_jsonlines(args.question_records_file)

    languages_to_use = set(args.languages_list)

    for qnum, qa in enumerate(tqdm(qas)):
        if args.question_records_file is not None:
            # sanity check
            assert(qa['answers'] == q_recs[qnum]['answers'])
            if q_recs[qnum]['lang'] not in languages_to_use:
                continue

        q=qa['question']
        p=qa['positive_ctxs']
        n=qa['negative_ctxs']
        hn=qa['hard_negative_ctxs']

        en_q = question_translations[q]

        if args.do_not_run_ir:
            continue

        retrieved = run_query(en_q, searcher, args.max_num_ir_based_negatives * 2) # assuming 50/50 positive/negative ratio

        negatives = []
        positives = []

        positive_ctxs_texts = [ctxt['text'] for ctxt in p]

        for hit in retrieved:
            #string_to_search = hit[2][1] if args.do_not_run_match_in_title else hit[2][0] + ' ' + hit[2][1]
            string_to_search = hit[3] if args.do_not_run_match_in_title else hit[2] + ' ' + hit[3]
            if answer_checker.has_answer(qa['answers'], string_to_search):
                if len(positives) < args.max_num_ir_based_positives and not max([ ctxt in hit[3] for ctxt in positive_ctxs_texts ]):
                    #positives.append({'title': hit[2][0], 'text': hit[2][1]})
                    positives.append({'title': hit[2], 'text': hit[3]})
            else:
                if len(negatives) < args.max_num_ir_based_negatives:
                    # negatives.append({'title': hit[2][0], 'text': hit[2][1]})
                    negatives.append({'title': hit[2], 'text': hit[3]})

            if len(positives) >= args.max_num_ir_based_positives and len(negatives) >= args.max_num_ir_based_negatives:
                break

        qa['ir_negative_ctxs'] = negatives
        qa['ir_positive_ctxs'] = positives

        print('adding ' + str(len(qa['ir_negative_ctxs'])) + ' IR based negatives, ' + str(len(qa['ir_positive_ctxs'])) + ' IR based positives')

        if qnum > 0 and (qnum % 1000) == 0:
            output_file = args.output_file.replace('.json', '_' + str(qnum) + '.json')
            print('writing ', output_file)
            with open(output_file, 'w') as out_f:
                json.dump(qas, out_f, indent=4)

    output_file = args.output_file
    print('writing ', output_file)
    with open(output_file, 'w') as out_f:
        json.dump(qas, out_f, indent=4)


# do main
if __name__=='__main__':
   main()
