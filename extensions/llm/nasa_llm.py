

from primeqa.components.retriever.dense import ColBERTRetriever
from primeqa.components.reader.prompt import PromptGPTReader
from primeqa.pipelines.qa_pipeline import QAPipeline
import json

# load questions

nasa_question_file = "/dccstor/phalanx/zhrong/largelm/nasa_data/es_qa_sq2format_val.v2.json"

with open(nasa_question_file) as f:
    nasa_data = json.load(f)['data']

# setup ColBERT index
index_root = "/dccstor/colbert-ir/franzm/indexes/mar3_6_10/mar3_6_10_/indexes/"
index_name = "mar3_6_10_indname"
collection = "/dccstor/colbert-ir/franzm/data/nasa/documents.tsv"


retriever = ColBERTRetriever(index_root = index_root, 
                                     index_name = index_name, 
                                     collection = collection, 
                                     max_num_documents = 3)
retriever.load()

from primeqa.components.reader.prompt import PromptFLANT5Reader
reader = PromptFLANT5Reader(model_name="google/flan-t5-large")
reader.load()

# setup the pipeline
pipeline = QAPipeline(retriever, reader)

for document in nasa_data:
    for paragraph in document['paragraphs']:
        for questions in paragraph['qas']:
            print(questions['question'])
            print(questions['answers'])
            questions = [questions['question']]
            prompt_prefix = "Answer the following question after looking at the text."
            answers = pipeline.run(questions,prefix=prompt_prefix)
            print(json.dumps(answers, indent=4))