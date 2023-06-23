from typing import List
from tqdm import tqdm

from primeqa.components.base import Reader, Retriever
from primeqa.components.retriever.searchable_corpus import SearchableCorpus
from primeqa.components.reader import GenerativeReader

class QAPipeline:
    def __init__(self, retriever: Retriever, reader: Reader) -> None:
        self.retriever = retriever
        self.reader = reader
        self.corpus_passages = []
        with open(self.retriever.collection, "r") as infile:
            for line in tqdm(infile):
                id, text, title = line.split("\t")
                self.corpus_passages.append(title + " " + text)

    def run(self, input_texts: List[str], prefix="", suffix="", use_retriever=True):
        contexts = []
        if use_retriever:
            search_results = self.retriever.predict(input_texts=input_texts)
            for result in search_results:
                context = [self.corpus_passages[int(p[0])] for p in result]
                contexts.append(context)

        reader_answers = self.reader.predict(
            input_texts, contexts, prefix=prefix, suffix=suffix
        )
        result = {}
        for i, answers_i in reader_answers.items():
            i_result = {}
            i_result["answers"] = answers_i
            if use_retriever:
                i_result["passages"] = contexts[int(i)]
            result[i] = i_result

        return result

class RAG:
    def __init__(self, retriever: SearchableCorpus, reader: GenerativeReader) -> None:
        self.retriever = retriever
        self.reader = reader
       

    def run(self, questions: List[str], prefix="", suffix="Answer"):
        
        _,search_results = self.retriever.search(questions)
        contexts = [sr['texts'] for sr in search_results]
    
        reader_answers = self.reader.predict(
            questions, contexts, prefix=prefix, suffix=suffix
        )
        
        result = []
        for i, answers_i in reader_answers.items():
            i_result = {}
            i_result['question'] = questions[int(i)]
            i_result["answer"] = answers_i[0]['span_answer_text']
            i_result["passages"] = ["Passage: "+c[:200]+"..." for c in contexts[int(i)][:3]]
            result.append(i_result)

        return result