import os
import json
import csv
from csv import DictReader
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def load_queries(queries_tsv_filepath):
    queries = {}
    with open(queries_tsv_filepath,'r') as f:
        reader = DictReader(f,delimiter='\t',fieldnames=['id','text'])
        for row in reader:
            queries[row['id']] = row['text']
    return queries

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