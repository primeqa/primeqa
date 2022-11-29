import os
import sys
import json
from typing import List
from operator import itemgetter
import argparse
import logging
from transformers import HfArgumentParser
from discovery import WatsonDiscoveryRetriever
from primeqa.pipelines.components.retriever.dense import ColBERTRetriever
from primeqa.pipelines.components.reader.extractive import ExtractiveReader


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SUPPORTED_READERS = {
    ExtractiveReader.__name__: ExtractiveReader
}

SUPPORTED_RETRIEVERS = {
    WatsonDiscoveryRetriever.__name__: WatsonDiscoveryRetriever,
    ColBERTRetriever.__name__: ColBERTRetriever
}

# required to get normalized reader scores
MIN_NUM_ANSWERS_FROM_READER = 10
IR_WEIGHT=0.7
MAX_NUM_DOCUMENTS=5


def handle_args():
    usage = "Run Open Retrieval QA pipeline"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument("--config_file",type=str,required=True,
        help="config json file containing retriever and reader parameters",
    )
    parser.add_argument(
        "--query_file", type=str, required=True, help="queries tsv file id\tquery"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="directory for output files"
    )
    args = parser.parse_args()
    logger.info(json.dumps(vars(args),indent=2))
    return args

class OpenRetrievalQA:
    
    def __init__(self, retriever, reader, ir_weight, corpus_tsv_file=None) -> None:
        self.retriever = retriever
        self.reader = reader
        self.ir_weight = ir_weight
        self.corpus_tsv_file = corpus_tsv_file
        
        if retriever.__class__.__name__ not in SUPPORTED_RETRIEVERS:
            raise ValueError(f"Retriever of type {retriever.__class__.__name__} is not supported.")
        
        if retriever.__class__.__name__ == ColBERTRetriever.__name__:
            if self.corpus_tsv_file is None:
                raise ValueError(f"Corpus tsv required when using ColBERTRetriever")
            self._corpus = self._load_corpus(corpus_tsv_file)
            
        
    def _load_corpus(self, filepath):
        with open(filepath,'r') as f:
            return f.readlines()
        
    
    def _min_max(self,scores):
        low = min(scores)
        high = max(scores)
        width = high - low
        if width == 0:
            raise RuntimeError("Division by zero")
        return [(s - low) / width for s in scores]

    def _normalize(self, hits: list, raw_field, norm_field):
        for idx, confidence in enumerate(self._min_max([hit[raw_field] for hit in hits])):
            hits[idx][norm_field] = confidence
            
    def _combine_ir_mrc_scores(self, answers):
    
        for answer in answers:
            answer['score'] = answer['ir_score']*self.ir_weight + (1-self.ir_weight)* answer['mrc_score']
        sorted_answers = sorted(answers,key=itemgetter('score'),reverse=True)
        return sorted_answers

    def _retrieve_with_discovery_retriever(self, queries):
        # self.retriever.max_num_documents = self.retriever.max_num_documents
        return self.retriever.retrieve(queries)


    def _retrieve_with_colbert_retriever(self, queries):
        # self.retriever.max_num_documents = max_num_documents
        
        queries_hits =  self.retriever.retrieve(queries)
        
        results = []
        for query_hits in queries_hits:
            hits = [
                {
                    "text": self._corpus[hit[0]].strip().split('\t')[1],
                    "search_score": hit[1],
                    "document_id": hit[0],
                    "title": self._corpus[hit[0]].strip().split('\t')[2] if len(self._corpus[hit[0]].strip().split('\t')) == 3 else None
                }
                for hit in query_hits
            ]
            results.append(hits)
        return results

    def _rank_answers(self, answers):
        # normalize ir score
        self._normalize(answers, 'search_score', 'ir_score' )
        # normalize reader scores
        self._normalize(answers, 'span_answer_score', 'mrc_score' )
        # score combination
        sorted_answers = self._combine_ir_mrc_scores(answers)
        return sorted_answers

    def _find_answers(self, query, hits):
        
        for hit in hits:
            # only one query
            answers = self.reader.apply([query], [[hit['text']]])[0]
            # we only consider one answer per hit
            top_answer = answers[0]
            for key in top_answer:
                hit[key]  = top_answer[key]
                
        ranked_answers = self._rank_answers( hits )
        return ranked_answers
    
    
    @staticmethod
    def get_extractive_reader(model_name_or_path, max_answer_length, max_num_answers):
        # MIN_NUM_ANSWERS_FROM_READER is need to get scaled span answer scores
        max_num_answers = MIN_NUM_ANSWERS_FROM_READER if max_num_answers < MIN_NUM_ANSWERS_FROM_READER else max_num_answers
        reader = ExtractiveReader(model=model_name_or_path,max_answer_length=max_answer_length,max_num_answers=max_num_answers)
        reader.load()
        return reader


    @staticmethod
    def get_colbert_retriever(index_root: str, 
                            index_name: str, 
                            checkpoint: str, 
                            max_num_documents: int = 5 ):
        retriever = ColBERTRetriever(index_root=index_root,
                                    index_name=index_name,
                                    checkpoint=checkpoint,
                                    max_num_documents=max_num_documents)
        retriever.load()
        return retriever
        
        
    @staticmethod
    def get_discovery_retriever(endpoint: str, 
            api_key: str, 
            project_id: str, 
            index_name: str,
            max_num_documents: int = MAX_NUM_DOCUMENTS):
        retriever = WatsonDiscoveryRetriever(index_root="", index_name=index_name,
                                        endpoint=endpoint,
                                        api_key=api_key,
                                        project_id=project_id,
                                        max_num_documents=max_num_documents
                                        )
        retriever.load()
        return retriever   
    
    @staticmethod
    def get_instance(config_json_file: str):
        
        with open(config_json_file,'r') as f:
            config = json.load(f)
            
        # only the first till ensembling is supported
        retriever_config = config['retrievers'][0]
        retriever = None
        corpus_file_path = None
        if retriever_config['name'] == ColBERTRetriever.__name__:
            retriever = OpenRetrievalQA.get_colbert_retriever(index_root=retriever_config['index_root'],
                                                  index_name=retriever_config['index_name'], 
                                                  checkpoint=retriever_config['checkpoint'], 
                                                  max_num_documents=retriever_config['max_num_documents'])
            corpus_file_path = retriever_config['corpus_tsv_file_path']
                                                  
        elif retriever_config['name'] == WatsonDiscoveryRetriever.__name__:
            retriever = OpenRetrievalQA.get_discovery_retriever(endpoint=retriever_config['endpoint'],
                                                                api_key=retriever_config["apikey"],
                                                                project_id=retriever_config["project_id"],
                                                                index_name=retriever_config["index_name"],
                                                                max_num_documents=retriever_config["max_num_documents"]
                                                             )
        else:
            raise ValueError(f"Unsupported retriever {retriever_config['name']} ")
        
        reader_config = config['reader']
        reader = OpenRetrievalQA.get_extractive_reader(reader_config['model_name_or_path'], 
                                               reader_config['max_answer_length'], 
                                               reader_config['max_num_answers'])
        
        ir_weight = float(config['reranking']["ir_weight"])
        return OpenRetrievalQA(retriever, reader, ir_weight, corpus_file_path)
    
    def ask(self, query):
        if self.retriever.__class__.__name__ is ColBERTRetriever.__name__:
            hits = self._retrieve_with_colbert_retriever( [query] )
            
        elif self.retriever.__class__.__name__  is WatsonDiscoveryRetriever.__name__:
            hits = self._retrieve_with_discovery_retriever( [query]
                                                        )
        answers =  self._find_answers(query, hits[0])
        return answers

def format_text(text, start, end):
    mark_text = text[start:end]
    if mark_text.strip().endswith('</em>'):
        print(mark_text)
    formatted_text = text[0:start] + '<answer>' + text[start:end] + '</answer>' + text[end:]
    return formatted_text


def save_predictions(query_to_answers, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_file = os.path.join(output_dir,"predictions.json")
    with open(output_file,'w') as f:
        json.dump(query_to_answers,f,indent=2)
    logger.info(f"Wrote {output_file}")
    
    output_file = os.path.join(output_dir,"predictions_processed.json")
    predictions_processed = {}
    for id, answers in query_to_answers.items():
        predictions_processed[id] = answers[0]['span_answer_text']        
    with open(output_file,'w') as f:
        json.dump(predictions_processed,f,indent=2)
    logger.info(f"Wrote {output_file}")
    
    output_file = os.path.join(output_dir,"predictions_inline.json")
    jsonlines = []
    for id, answers in query_to_answers.items():
        answerdict = {
            'id': id,
            'question': answers[0]['question'],
        }
        for i, answer in enumerate(answers[0:3]):
            answerdict[f"rank_{i}_answer"] = answers[i]['span_answer_text']
            answerdict[f"rank_{i}_document"] = format_text(answers[i]['text'],answers[i]['span_answer']['start_position'],answers[i]['span_answer']['end_position'])
        jsonlines.append(answerdict)
    with open(output_file,'w',encoding='utf-8') as f:
        json.dump(jsonlines,f, indent=4,ensure_ascii=False)
    print('Wrote', output_file)
            
def load_queries(queries_file):
    id_to_query = {}
    with open(queries_file,'r') as f:
        for line in f:
            id,text = line.strip().split('\t')
            id_to_query[id] = text
    return id_to_query
    
     
def main():    
    
    args = handle_args()
    queries_file = args.query_file
    config_file = args.config_file
    output_dir = args.output_dir
        
    queries = load_queries(queries_file)
    
    orqa = OpenRetrievalQA.get_instance(config_file)
    
    query_to_answers = {}
    
    for id, question in queries.items():
        answers = orqa.ask(question)
        for answer in answers:
            answer["id"] = id
            answer["question"] = question
        query_to_answers[id] = answers
        
    save_predictions(query_to_answers, output_dir)
        
        
if __name__ == "__main__":
    main()
