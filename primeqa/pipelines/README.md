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

Here we provide some simple to use interfaces that perform processing steps such as indexing, searching and answer extraction from text.  This allows the components to be used as building blocks and allows switching out alternative implementations.  
The interfaces for the retrieval, indexing and reader components are defined [here](./components/base.py)

## Retrieval Components

### ColBERT Engine

The following examples show how to use the _Component_ interface to index and search using the ColBERT Engine.

There is a corresponding Jupyter notebook [here](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir_pipeline.ipynb).  The notebook also shows examples of data files and their formats.

#### Indexing with ColBERT Engine

- **Initializing Indexer**
```python
from primeqa.components.indexer.dense import ColBERTIndexer

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
from primeqa.components.retriever.dense import ColBERTRetriever

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
from primeqa.components.reader.extractive import ExtractiveReader

reader = ExtractiveReader("PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110")
reader.load()

```
- Step 2: Execute the reader in inference mode:
```python
question = ["Which country is Canberra located in?"]
context = [["""Canberra is the capital city of Australia. 
Founded following the federation of the colonies of Australia 
as the seat of government for the new nation, it is Australia's 
largest inland city"""]]
answers = reader.predict(question,context)  
print(json.dumps(answers, indent=4))  
```


### Generative FiD Reader

A Generative Reader takes a question and uses a set of supporting passages to generate an answer. In contrast to the Extractive Reader, where the answers are usually short spans extracted from the input passages, the Generative Reader generates complex, multi-sentence answers.

PrimeQA implements a [Fusion In Decoder(FiD)](https://arxiv.org/abs/2007.01282) generative reader. 

Follow the steps below to use the `GenerativeFiDReader`:

- Step 1:  Initialize the reader.
```python
import json
from primeqa.components.reader.generative import GenerativeFiDReader
fid_reader = GenerativeFiDReader()
fid_reader.load()
```

- Step 2: Execute the reader in inference mode:
```python
question = ["What causes the trail behind jets at high altitude?"]
context = [["""Chemtrail conspiracy theory The chemtrail conspiracy theory is based 
            on the erroneous belief that long-lasting condensation trails are 
            \"chemtrails\" consisting of chemical or biological agents left in the 
            sky by high-flying aircraft, sprayed for nefarious purposes undisclosed 
            to the general public. Believers in this conspiracy theory say that while 
            normal contrails dissipate relatively quickly, contrails that linger must 
            contain additional substances. Those who subscribe to the theory speculate 
            that the purpose of the chemical release may be solar radiation management,
            weather modification, psychological manipulation, human population control, 
            or biological or chemical warfare, and that the trails are causing 
            respiratory illnesses""",
            """Associated with jet streams is a phenomenon known as clear-air turbulence 
            (CAT), caused by vertical and horizontal wind shear caused by jet streams. 
            The CAT is strongest on the cold air side of the jet, next to and just under 
            the axis of the jet. Clear-air turbulence can cause aircraft to plunge and so 
            present a passenger safety hazard that has caused fatal accidents, such as the 
            death of one passenger on United Airlines Flight 826. 
            Section: Uses.:Possible future power generation.""",
            """Contrails are a manmade type of cirrus cloud formed when water vapor from 
            the exhaust of a jet engine condenses on particles, which come from either the 
            surrounding air or the exhaust itself, and freezes, leaving behind a visible trail. 
            The exhaust can also trigger the formation of cirrus by providing ice nuclei 
            when there is an insufficient naturally-occurring supply in the atmosphere. 
            One of the environmental impacts of aviation is that persistent contrails can 
            form into large mats of cirrus, and increased air traffic has been implicated 
            as one possible cause of the increasing frequency and amount of cirrus"""]]
answers = fid_reader.predict(question,context)  
print(json.dumps(answers, indent=4)) 
```

## QA Pipeline

The QA pipeline is used for Open Retrieval Question Answering. Open retrieval systems query large document stores for relevant passages. Long Form Question Answering (LFQA) is a generative task where the retrieved passages are used to generate a complex multi-sentence answer.

In this example we show a QA Pipeline using a ColBERT retriever and a Fusion in Decoder (FID) generator.

Instructions to create a ColBERT index and an FiD model for KILT-ELI5 can be found [here](https://github.com/primeqa/primeqa/blob/main/examples/lfqa/README.md)

- Step 1:  Initialize the retriever.

```python
retriever = ColBERTRetriever(index_root = index_root, index_name = index_name, collection = collection, max_num_documents = 3)
retriever.load()
```

- Step 2:  Initialize the reader model. 

```python
reader = GenerativeFiDReader(model_name_or_path = model)
reader.load()
```

- Step 3:  Initialize the QA pipeline. 

```python
lfqa_pipeline = QAPipeline(retriever, reader)
```

- Step 4:  Execute the LFQA pipeline in inference mode. 

```python
queries=["What causes the trail behind jets at high altitude?"]
answers = lfqa_pipeline.run(query)
```