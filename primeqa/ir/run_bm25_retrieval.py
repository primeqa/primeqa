import os
import argparse
import csv
import json
from typing import List, Dict
from primeqa.ir.sparse.retriever import PyseriniRetriever
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def handle_args():
    parser = argparse.ArgumentParser(description='Run BM25 search'
)
    parser.add_argument('--index_path', type=str,required=True,help="Path to a lucene index")
    parser.add_argument('--queries_file', type=str, required=True, help='Path to queries file in ColBERT tsv [id\tquery] format')
    parser.add_argument('--top_k', type=int, required=False, default=1000, help='Number of documents/passages to retrieve')
    parser.add_argument('--xorqa_data_file', type=str, required=False, default=None, help="Required to extract language id for XORQA evaluation." )
    parser.add_argument('--max_hits_to_output', type=int, required=False, default=100, help='Number of hits to output for XORQA evaluation.')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    args=parser.parse_args()
    return args
 
def load_queries(filepath: str) -> Dict:
    id_to_question = {}
    with open(filepath,'r') as f:
        for line in f:
            id, question = line.strip().split('\t')
            id_to_question[id] = question
    return id_to_question

def run_search(index_path: str, queries: List, top_n: int = 1000, k1: float = 0.9, b: float = 0.4) -> Dict:
    logger.info(f"Loading index {index_path}")
    searcher = PyseriniRetriever(index_path, use_bm25=True, k1=k1, b=b)
    logger.info(f"Running search")
    id_to_hits = {}
    for id in queries:
        question = queries[id]
        logger.info(f'Running search: {id} {question}')
        hits = searcher.retrieve(question,top_n)
        id_to_hits[id] = hits
    return id_to_hits

def write_colbert_ranking_tsv(output_dir: str , id_to_hits: Dict):
    output_file = os.path.join(output_dir,'ranking.tsv')
    search_results = []
    for id in id_to_hits:
        for i, hit in enumerate(id_to_hits[id]):
            result = {
                "id": id,
                "docid": hit['doc_id'],
                "rank": i+1, 
                "score": hit['score']
            }
            search_results.append(result)

    with open(output_file,'w',encoding='utf-8') as f:
        writer = csv.writer(f, delimiter='\t', lineterminator='\n')
        for r in search_results:
            writer.writerow(r.values())
    logger.info(f"Wrote {output_file}")
        
def get_language_id(xorqa_data_file: str) -> Dict:
    id_to_lang = {}
    with open(xorqa_data_file,'r') as f:
        for line in f:
            data = json.loads(line.strip())
            id_to_lang[data['id']] = data['lang']
    return id_to_lang

def write_xorqa_json(output_dir: str, id_to_hits: Dict, top_n: int =100, xorqa_data_file: str = None):
    id_to_lang = None
    if xorqa_data_file != None:
        id_to_lang = get_language_id(xorqa_data_file)

    output_file = os.path.join(output_dir,'ranking_xortydi_format.json')
    search_results = []
    for id in id_to_hits:
        json_data = {
            "id" : id,
            "lang" : id_to_lang[id] if id_to_lang != None else "", 
            "ctxs" : []
        }
        for hit in id_to_hits[id][:top_n]:
            json_data["ctxs"].append(hit['text'])
        search_results.append(json_data)
    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(search_results,f, indent=4)
    logger.info(f"Wrote {output_file}")

def write_search_results(id_to_hits: Dict, output_dir: str, max_hits_output: int, xorqa_data_file: str =None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    write_xorqa_json(output_dir, id_to_hits, top_n=max_hits_output, xorqa_data_file=xorqa_data_file)
    write_colbert_ranking_tsv(output_dir, id_to_hits)

def main():
    args = handle_args()
    logger.info("starting")
    id_to_questions = load_queries(args.queries_file)
    id_to_hits = run_search(args.index_path, id_to_questions, top_n=1000)
    write_search_results(id_to_hits, args.output_dir, args.max_hits_to_output, args.xorqa_data_file)

if __name__ == '__main__':
    main()
    logger.info("Success...")