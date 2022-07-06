#!/bin/env python
# -*- coding: utf-8 -*-

import argparse
import logging
import pandas as pd
import json

logger=logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)


def handle_args():
    usage='usage'	
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--corpus','-c', required=True)
    parser.add_argument('--queries','-q', required=True)
    parser.add_argument('--pred', '-p', required=True)
    parser.add_argument('--out','-o', required=True)  # prefix
    parser.add_argument('--topn', '-n', type=int, default=50)  # restrict output list of contexts to top n
    parser.add_argument('--positional_qids_in_predictions', '-P', action="store_true", default=False) # the predictions file has 0, 1, 2 ..., as opposed to actual query IDs such as -"1018987027025691298"

    args=parser.parse_args()
    return args


def yield_json_lines(srcFn):
    with open(srcFn) as srcIn:
        for q in srcIn:
            yield json.loads(q)



#
# the corpus file
#
# reference
#   --corpus collection.tsv \
# has id \t text \t title with header
def read_corpus(ctx_file):
    logger.info('handle_corpus')
    dfd=pd.read_csv(ctx_file, sep='\t')
    return dfd.set_index('id')

#
# queries are --queries queries.tsv
# id \t question no header
def read_queries(q_file):
    logger.info('handle_queries')
    dfq=pd.DataFrame.from_records(yield_json_lines(q_file)).reset_index()
    return dfq.set_index('index')


#
#
#
def read_colbert_results(pred_fn):
    dfp=pd.read_csv(pred_fn,sep='\t',names=['qid','docid','rank','rel'])
    return dfp.sort_values(['qid','rank'])



def main():
    args=handle_args()

    dfd=read_corpus(args.corpus)
    dfq=read_queries(args.queries)
    dfp=read_colbert_results(args.pred)

    if not args.positional_qids_in_predictions:
        qid_to_dfq_pos = {}
        for q_pos in list(range(len(dfq))):
            qid_to_dfq_pos[int(dfq.loc[q_pos]['id'])] = q_pos

    output_list=[]
    for qnum,dfpq in dfp.groupby('qid'):
        if not args.positional_qids_in_predictions:
            qnum = qid_to_dfq_pos[qnum]

        query_row=dfq.loc[qnum]
        query_row['id']
        query_row['lang']
        ctxs=[item for item in dfpq['docid'].head(args.topn).map(dfd['text']).tolist() if isinstance(item, str)]
        for rank, docid in enumerate(dfpq['docid'].head(args.topn)):
            if not docid in dfd['text']:
                print('WARNING: docid ' + str(docid) + ' , found in query ' + str(qnum) + ' rank ' + str(rank) + ' not found in corpus ' + args.corpus)
        output_list.append({'id':query_row['id'],
                           'lang':query_row['lang'],
                           'ctxs':ctxs
                       })

    with open(args.out,'wt') as json_out:
        json.dump(output_list, json_out, ensure_ascii=False, indent=2)


# do main
if __name__=='__main__':
   main()
