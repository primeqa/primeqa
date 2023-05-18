import os
from tqdm import tqdm
import csv
from dataclasses import dataclass, field
from transformers import HfArgumentParser

from primeqa.pipelines.retrieve_rerank import RetrieveRerankPipeline
from primeqa.components.retriever.sparse import BM25Retriever
from primeqa.components.reranker.colbert_reranker import ColBERTReranker


queries = [
    "Which is the largest city in the state of Oregon?",
    "In which country was cholera first diagnosed?"
]
index_root = "/dccstor/colbert-ir/bsiyer/PQLL/indexes/bm25"
index_name = "wikiindex"
reranker_model = "/dccstor/colbert-ir/bsiyer/PQLL/experiments/apr5_2023/2023-04/05/11.05.23/checkpoints/colbert.dnn.batch_17524.model"
reranker_model = "/dccstor/colbert-ir/franzm/experiments/mar3_22_21/2023-03/22/14.39.40/checkpoints/colbert-LAST.dnn"

drdecr_teacher = "/dccstor/colbert-ir/yulongl/colbertV2_experiments/distill_XOR_resume/6e-6/none/2022-08/13/00.25.53/checkpoints/colbert.dnn.batch_87618.model"

@dataclass
class RerankArguments():
    search_results: str = field(default=None, metadata={"help":"Path to the search results tsv file in format 'id\tdocid\trank\tscore"})

    queries: str = field(default=None, metadata={"help":"Path to the tsv file where each line is in format 'id\tquery'"})
    
    collection: str = field(default=None, metadata={"help":"Path to a corpus tsv in format 'id\ttext\ttitle"})
    
    include_title: bool = field(default=False, metadata={"help":"Prepend title to passage"})
    
    resume: bool = field(default=False, metadata={"help":"Resume from last processed query"})
    
    overwrite: bool = field(default=False, metadata={"help":"Overwrite reranked results file"})
    
    checkpoint: str = field(default=None, metadata={"help":"Path to a ColBERT checkpoint"})
    
    q_max_len: int = field(default=32, metadata={"help":"Query max length"})
    
    d_max_len: int = field(default=180, metadata={"help":"Document max length"})
    
    topk_to_rerank: int = field(default=None, metadata={"help":"Num hits to rerank"})

    output_file: str = field(default=None, metadata={"help":"Output file to write reranked results"})


def load_queries(in_file):
    print(f"Loading queries {in_file} ...")
    id_to_question = {}
    with open(in_file, 'r')  as f:
        csv_reader = csv.DictReader(f, fieldnames=["id", "question"], delimiter="\t")
        for row in csv_reader:
            id_to_question[row['id']]  = row['question']
    return id_to_question
        

def load_corpus(in_file, include_title=False):
    print(f"Loading collection {in_file} ...")
    with open(in_file) as f:
        lines = f.readlines()
    id_to_passage = {}
    id_to_title = {}
    with open(in_file) as f:
        csv_reader = csv.DictReader(f, fieldnames=["id", "text", "title"], delimiter="\t")
        for row in tqdm(csv_reader, total=len(lines)):
            id_to_passage[row['id'] ] = row['text']
            if include_title:
                id_to_title[row['id'] ] = row['title']

    return id_to_passage, id_to_title
    

def load_search_results(in_file, num_lines=1000000):
    print(f"Loading search results {in_file} ...")
    with open(in_file) as f:
        lines = f.readlines()
        num_lines=len(lines)
    qid_to_hits = {}
    with open(in_file) as f:
        csv_reader = csv.DictReader(f, fieldnames=["qid", "docid", "rank", "score"], delimiter="\t")
        for row in tqdm(csv_reader, total=num_lines):
            if row['qid'] not in qid_to_hits:
                qid_to_hits[row['qid']] = []
            qid_to_hits[row['qid']].append(row['docid'])
    return qid_to_hits
            
def rerank(output_file, resume, reranker, qid_to_question, id_to_passages,id_to_title, qid_to_hits,topk_to_rerank=None, include_title=False):
    print("Reranking...")
    print("Num queries:", len(qid_to_hits))
    qid_to_reranked_results = {}
    
    
    if resume:
        raise RuntimeError("resume not implemented")
    
    with open(output_file,'w') as f:
        for qid in tqdm(qid_to_hits):
            hits = []
            #print(f"{qid}\n {len( qid_to_hits[qid])}")
            for docid in qid_to_hits[qid]:
                hit = {
                    "document": {
                        "document_id": docid,
                        "title": id_to_title[docid] if include_title else "",
                        "text": id_to_passages[docid]
                    }
                }
                hits.append(hit)
            hits = hits[0:topk_to_rerank] if topk_to_rerank is not None else hits
            if len(hits) < topk_to_rerank:
                print(f"{qid} has fewer hits than requested topk: {topk_to_rerank} hits:{len(hits)}")
            reranked_results = reranker.predict([qid_to_question[qid]], [hits], include_title=include_title,  max_num_documents=len(hits))
            # print(f"len reranked_results {len(reranked_results)}")
            # print(f"Reranked {qid}\n {len(reranked_results[0])}")
            assert len(reranked_results[0]) ==  len(qid_to_hits[qid][0:topk_to_rerank])
            qid_to_reranked_results[qid] = reranked_results[0]
            if len(qid_to_reranked_results) == 1000:
                lines = format_output(qid_to_reranked_results)
                print(f"Writing {len(lines)}")
                f.writelines([f'{l}\n' for l in lines])
                f.flush()
                qid_to_reranked_results = {}
                
        if len(qid_to_reranked_results) > 0:
            lines = format_output(qid_to_reranked_results)
            print(f"Writing {len(lines)}")
            f.writelines([f'{l}\n' for l in lines])
            f.flush()
        print(f"Wrote {output_file}")
        
                
def get_colbert_reranker(checkpoint, q_max_len=32, d_max_len=180):
    print(f"Loading ColBERT model q_max_len:{q_max_len} d_max_len:{d_max_len} checkpoint:{checkpoint} ...")
    reranker = ColBERTReranker(checkpoint, query_maxlen=q_max_len, doc_maxlen=d_max_len)
    reranker.load()
    return reranker

def format_output(qid_to_reranked_results):
    lines = []
    for qid in qid_to_reranked_results:
        reranked_results = qid_to_reranked_results[qid]
        for i, r in enumerate(reranked_results):
            lines.append(f"{qid}\t{r['document']['document_id']}\t{i+1}\t{r['score']}")
    return lines
    

def write(output_file, qid_to_reranked_results):
    lines = []
    for qid in qid_to_reranked_results:
        reranked_results = qid_to_reranked_results[qid]
        for i, r in enumerate(reranked_results):
            lines.append(f"{qid}\t{r['document']['document_id']}\t{i+1}\t{r['score']}")
    print(f"Writing {len(lines)}")
    with open(output_file,'w') as f:
        f.writelines([f'{l}\n' for l in lines])
    print("Wrote", output_file)
            
            
    
def main():
    parser = HfArgumentParser([RerankArguments])
    (rerank_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    if os.path.exists(rerank_args.output_file) and not rerank_args.overwrite and not rerank_args.resume:
        raise ValueError(
                f"Output file ({rerank_args.output_file}) already exists. "
                "Use --overwrite to overcome."
            )
        
    qid_to_question = load_queries(rerank_args.queries)
    id_to_passage, id_to_title = load_corpus(rerank_args.collection, include_title=rerank_args.include_title)
    qid_to_hits = load_search_results(rerank_args.search_results)
    
    reranker = get_colbert_reranker(rerank_args.checkpoint,q_max_len=rerank_args.q_max_len, d_max_len=rerank_args.d_max_len)
    rerank(rerank_args.output_file, rerank_args.resume, reranker, qid_to_question, id_to_passage, id_to_title, qid_to_hits,include_title=rerank_args.include_title, topk_to_rerank=rerank_args.topk_to_rerank)
    # write(rerank_args.output_file, qid_to_reranked_results)
    
    print("Done...")
    
if __name__ == '__main__':
    main()
    
