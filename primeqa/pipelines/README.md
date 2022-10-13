<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.pipelines

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# Pipelines

## Retrieval Components

### ColBERT Engine

The following examples show how to use the _Pipelines_ interface to index and search using the ColBERT Engine.

There is a corresponding Jupyter notebook [here](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir_pipeline.ipynb).  The notebook also shows examples of data files and their formats.

#### Indexing with ColBERT Engine

- **Initializing Indexer**
```python
from primeqa.pipelines.components.indexer.dense import ColBERTIndexer

indexer = ColBERTIndexer(checkpoint = checkpoint_fn, index_root = index_root, index_name = index_name, num_partitions_max = 2)
indexer.load()
```
The `checkpoint_fn` variable points to an existing model (checkpoint), more details on model training are in the Jupyter notebook [here](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir.ipynb). 

- **Indexing data collection**
```python
indexer.index(collection = collection_fn)
```
The `collection_fn` variable points to a data collection to be indexed. It is a _.tsv_ file, containing records in the form of _[ID, text, title]_ triples.
An example of a collection file is [here](https://github.com/primeqa/primeqa/blob/main/tests/resources/ir_dense/xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv).

This table shows the three lines from the file, with _text_ fields truncated:

| id | text | title |
|----|-------|-------|
| 1 | "The Kangxi Emperor's reign of 61 years ... | Kangxi Emperor |
| 2 | Yao. The Bamboo Annals says that when Emperor Zhuanxu died ... | Emperor Zhi |


#### Search with ColBERT Engine

- **Initializing Retriever**

```python
from primeqa.pipelines.components.retriever.dense import ColBERTRetriever

retriever = ColBERTRetriever(index_root = index_root, index_name = index_name, max_num_documents = 1)
retriever.load()
```
The `index_root` and `index_name` varibles are the same is in the indexing step.

The `max_num_documents` variable specifies the number of documents retrieved.

- **Searching with Retriever**

```python
results = retriever.retrieve(input_texts = ['Who is Michael Wigge'])
```
The `max_num_document` variable contains document_id and scores of the retrieved documents.

## Reader Components

### Extractive Reader

The Extractive Reader takes a question and a set of passages and returns an answer by extracting a span of text in the passages.
Follow the steps below to use the extractive reader:

- Step 1:  Initialize the reader. You can choose any of the MRC models we currently have [here](https://huggingface.co/PrimeQA).
```python
import json
from primeqa.pipelines.components.reader.extractive import ExtractiveReader
reader = ExtractiveReader("PrimeQA/tydiqa-primary-task-xlm-roberta-large")
```
- Step 2: Execute the reader in inference mode:
```python
question = ["Which country is Canberra located in?"]
context = ["""Canberra is the capital city of Australia. 
Founded following the federation of the colonies of Australia 
as the seat of government for the new nation, it is Australia's 
largest inland city"""]
answers = reader.apply(question,context)  
print(json.dumps(answers, indent=4))  
```