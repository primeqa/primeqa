#!/bin/env python
# -*- coding: utf-8 -*-

import csv
import argparse
import os
import pickle
import time
import random
import re
from tqdm.auto import tqdm

def handle_args():
    usage='usage'
    parser=argparse.ArgumentParser(usage)
    parser.add_argument('--input_passage_vectors', '-p', required=True)
    parser.add_argument('--input_query_vectors', '-q', required=True)
    parser.add_argument('--input_passages', '-s', required=True)
    parser.add_argument('--input_queries', '-r', required=True)

    parser.add_argument('--db_engine', '-e', default='pinecone', choices=['pinecone'], required=False)
    parser.add_argument('--output_ranks','-o', required=True)

    parser.add_argument('--num_embeddings_deleted', '-n', type=int, default=100, )
    parser.add_argument('--top_k', '-t', type=int, default=10, )

    parser.add_argument('--create_own_embeddings', '-w',default=False, action='store_true')
    parser.add_argument('--model_name', '-m', default='all-MiniLM-L6-v2')
    parser.add_argument('--dimension', '-d', type=int, default=768, )

    args=parser.parse_args()
    return args


def time_str(seconds: float) -> str:
    if seconds > 60 * 60:
        return f'{seconds/(60.0*60.0):.1f} hours'
    if seconds > 60:
        return f'{seconds/60.0:.1f} minutes'
    return f'{seconds:.1f} seconds'

def report_time(last_time):
    now = time.time()
    elapsed = now - last_time

    print(f'took {time_str(elapsed)}')
    return now

def main():
    random.seed(12345)

    last_time = time.time()

    # === read input files
    args=handle_args()

    with open(args.input_passage_vectors, 'rb') as in_file:
        passage_vectors = pickle.load(in_file)

    with open(args.input_query_vectors, 'rb') as in_file:
        query_vectors = pickle.load(in_file)

    input_passages = []  # {'text': text}
    with open(args.input_passages) as in_file:
        csv_reader = csv.DictReader(in_file, fieldnames=["id", "text", "title"], delimiter="\t")
        next(csv_reader)
        for row in csv_reader:
            assert len(row) == 3 or len(row) == 2, f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
            input_passages.append( {'text': ( row["title"] if len(row) == 3 else ' ' ) + ' ' + row["text"] } )

    input_queries = []  # {'text': text}
    with open(args.input_queries) as in_file:
        csv_reader = csv.reader(in_file, delimiter="\t")
        for row in csv_reader:
            assert len(row) == 2, f'Invalid .tsv record (has to contain 2 fields): {row}'
            input_queries.append( {'id': row[0], 'text': row[1]})


    print('=== done reading')
    last_time = report_time(last_time)

    # === initialize index and engine
    if args.db_engine == 'pinecone':
        import pinecone

        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        PINECONE_ENV = os.environ.get('PINECONE_ENV')

        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENV
        )

        index_name = (args.model_name + '__index').lower()
        index_name = re.sub('[^a-z0-9]', '-', index_name)

        #pinecone.delete_index('test-10k-v1')
        if index_name in pinecone.list_indexes():
            pinecone.delete_index(index_name)

        # only create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=args.dimension,
                metric='cosine'
            )

        # connect to the index
        index = pinecone.GRPCIndex(index_name)

    print('=== done initializing index')
    last_time = report_time(last_time)

    # === create embeddings
    if args.create_own_embeddings:
        if args.db_engine == 'pinecone':
            from sentence_transformers import SentenceTransformer
            import torch

            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if device != 'cuda':
                print(f"You are using {device}. This is much slower than using "
                      "a CUDA-enabled GPU. If on Colab you can change this by "
                      "clicking Runtime > Change runtime type > GPU.")

            model = SentenceTransformer(args.model_name, device=device)

        print('=== done initializing model')
        last_time = report_time(last_time)

        passage_vectors = []
        batch_size = 128
        for i in tqdm(range(0, len(input_passages), batch_size)):
            # find end of batch
            i_end = min(i+batch_size, len(input_passages))
            passage_vectors.extend(model.encode([passage['text'] for passage in input_passages[i:i_end]]))

        print('=== done creating passage embeddings')
        last_time = report_time(last_time)

    # === update index
    if args.db_engine == 'pinecone':
        batch_size = 128

        for i in tqdm(range(0, len(input_passages), batch_size)):
            # find end of batch
            i_end = min(i+batch_size, len(input_passages))
            # create IDs batch, IDs are just positions of the passages in input
            ids = [str(x) for x in range(i, i_end)]
            # create metadata batch
            metadatas = input_passages[i:i_end]
            # create embeddings
            vectors = passage_vectors[i:i_end]
            # create records list for upsert
            records = zip(ids, vectors, metadatas)
            # upsert to Pinecone
            index.upsert(vectors=records)

        # check number of records in the index
        index.describe_index_stats()

    print('=== done updating index')
    last_time = report_time(last_time)


    # === test index items deletion
    deleted_embedding_ids = [id for id in random.sample(range(len(input_passages)), args.num_embeddings_deleted)]

    if args.db_engine == 'pinecone':
        deleted_embedding_ids_str = [str(id) for id in deleted_embedding_ids]
        index.delete(ids=deleted_embedding_ids_str) # , namespace='example-namespace')

    print('=== done testing index items deletion')
    last_time = report_time(last_time)


    # === test index items insertion
    if args.db_engine == 'pinecone':
        vectors = [passage_vectors[id] for id in deleted_embedding_ids]
        metadatas = [input_passages[id] for id in deleted_embedding_ids]
        records = zip(deleted_embedding_ids_str, vectors, metadatas)
        index.upsert(vectors=records)

    print('=== done testing index items insertion')
    last_time = report_time(last_time)

    if args.create_own_embeddings:
        if args.db_engine == 'pinecone':
            query_vectors = []
            for query_number in range(len(input_queries)):
                query_vectors.append(model.encode(input_queries[query_number]['text']))

        print('=== done creating query embeddings')
        last_time = report_time(last_time)
    # === run retrieval
    out_ranks = []

    if args.db_engine == 'pinecone':
        for query_number in tqdm(range(len(query_vectors))):
        #for query_number in range(len(query_vectors)):
            response = index.query(query_vectors[query_number], top_k=args.top_k, include_metadata=True)
            for rank, match in enumerate(response['matches']):
                out_ranks.append([input_queries[query_number]['id'], match["id"], rank + 1, match["score"] ])

    print('=== done running retrieval')
    last_time = report_time(last_time)

    with open(args.output_ranks, 'w') as out_file:
        tsv_writer = csv.writer(out_file, quoting=csv.QUOTE_MINIMAL, lineterminator='\n', delimiter='\t')
        tsv_writer.writerows(out_ranks)

    if args.db_engine == 'pinecone':
        pinecone.delete_index(index_name)

# do main
if __name__=='__main__':
   main()
