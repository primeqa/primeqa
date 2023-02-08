from typing import List
from tqdm import tqdm

from primeqa.components.base import Reader, Retriever

class QAPipeline:
    def __init__(self, retriever: Retriever, reader: Reader) -> None:
    def __init__(self, retriever: Retriever, reader: Reader) -> None:
        self.retriever = retriever
        self.reader = reader
        self.corpus_passages = []
        with open(self.retriever.collection, 'r') as infile:
            for line in tqdm(infile):
                id,text,title = line.split('\t')
                self.corpus_passages.append(text)

    def run(self, input_texts: List[str]):
        search_results = self.retriever.retrieve(input_texts = input_texts)
        contexts = []
        for result in search_results:
            context = [self.corpus_passages[int(p[0])] for p in result]
            contexts.append(context)
        
        answers = self.reader.predict(input_texts,contexts)  
        return answers