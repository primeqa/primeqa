# Information Retrieval (IR)

Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

PrimeQA provides both dense and sparse IR components. 
- **Dense IR** is a ColBERT-based IR Engine enabling scalable BERT-based search
- **Sparse IR** is a Pyserini-based IR Engine enabling BM25 ranking using bag of words representation

This README shows how to run the basic model training, data indexing, retrieval using the `run_ir.py` script. 

Sample data files are [here](../../tests/resources/ir_dense), their formats are shown in the Jupyter notebooks [here](../../notebooks/ir/dense/dense_ir.ipynb) and [here](../../notebooks/ir/dense/dense_ir_student_teacher.ipynb). 

The steps involved in training a model using the DR.DECR (Dense Retrieval with Distillation-Enhanced Cross-Lingual Representation) Student/Teacher pipeline, as desribed in [Learning Cross-Lingual IR from an English Retriever](https://arxiv.org/abs/2112.08185), are outlined in the [Jupyter notebook](../../notebooks/ir/dense/dense_ir_student_teacher.ipynb).

The [Jupyter notebook](../../notebooks/ir/sparse/bm25_retrieval.ipynb) shows how to use the Sparse retriever API.


## Model Training

Dense IR requires training a model.  The following is an example of training a ColBERT model using the `run_ir.py` script.

The command uses training data in a _.tsv_ (tabulator character separated) file, containing training examples in the form of _[query, positive_passage, negative_passage]_ triples. An example of a training data file is [here](../../tests/resources/ir_dense/xorqa.train_ir_negs_5_poss_1_001pct_at_0pct.tsv).

This table shows two lines from the file, with the positive and negative passages truncated:

| query | positive_passage | negative_passage                                                           |
|-------|-------|----------------------------------------------------------------------------|
| 중국에서 가장 오랜기간 왕위를 유지한 인물은 누구인가? | "Kangxi Emperor The Kangxi Emperors reign of 61 years ... | Chiddy Bang new songs from the duo and in November 2009 debuted...         |
| 중국에서 가장 오랜기간 왕위를 유지한 인물은 누구인가? | Kangxi Emperor The Kangxi Emperors reign of 61 years ... | Emperor Zhi Yao. The Bamboo Annals says that when Emperor Zhuanxu died ... |

(English translation of the original Korean query is "_Who maintained the throne for the longest time in China?_")


```shell
python primeqa/ir/run_ir.py \
    --do_train \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --bsize 192 \
    --accum 6 \
    --maxsteps 100000 \
    --save_steps 20000
    --mask-punctuation \
    --lr 6e-06 \
    --similarity l2 \
    --model_type xlm-roberta-base \
    --triples <training_data> \
    --root <experiments_root_directory> \
    --experiment <experiment_label> 
```

The trained model is stored in `<experiments_root_directory>/<experiment_label>/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn`, with intermediate model files in the same `checkpoints` directory.
 
## Indexing

The following are examples of how to index a corpus using the `run_ir.py` script.
### Corpus Format
The command requires a corpus (collection) data in a _.tsv_ file, containing collection records in the form of _[ID, text, title]_ triples. The first line of the file contains a header record.
An example of a collection file is [here](../../tests/resources/ir_dense/xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv).

This table shows the three lines from the file, with _text_ fields truncated:

| id | text | title |
|----|-------|-------|
| 1 | "The Kangxi Emperor's reign of 61 years ... | Kangxi Emperor |
| 2 | Yao. The Bamboo Annals says that when Emperor Zhuanxu died ... | Emperor Zhi |

### Dense Index using ColBERT
Using a model trained as described [here](./README.md#model-training), the following command builds the index.

```shell
python primeqa/ir/run_ir.py \
    --do_index \
    --engine_type ColBERT \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 256 \
    --similarity l2 \
    --checkpoint <model_checkpoint> \
    --collection <document_collection> \
    --root <experiments_root_directory> \
    --experiment <experiment_label> \ 
    --index_name <index_label> \
    --compression_level 2
```
The index is stored in `<experiments_root_directory>/<experiment_label>/<index_label>` directory.

### Sparse Index using Pyserini

The following command builds an index for BM25 retrieval.  

```shell
python primeqa/ir/run_ir.py \
    --do_index \
    --engine_type BM25 \
    --corpus_path <document_collection> \
    --index_path <index_dir>
    --threads <num_threads>
```

## Retrieval

The command uses queries (questions) in a _.tsv_ file in the form of _[ID, text]_ records.
An example of a queries data file is [here](../../tests/resources/ir_dense/xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv).

### Dense Index Retrieval
The command uses a model and index as created in the previous two steps
```shell
python primeqa/ir/run_ir.py \
    --do_search \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 16 \
    --similarity l2 \
    --retrieve_only \
    --queries <query_file> \
    --checkpoint <model_checkpoint> \
    --collection <document_collection> \
    --root <experiments_root_directory> \
    --experiment <experiment_label> \ 
    --index_name <index_label> \
    --index_name ${EXPT}_indname \
    --ranks_fn <scores_and_ranks> \
    --nprobe 4
```

The resulting .tsv file, containing query IDs, document IDs, ranks, and scores is stored in `<scores_and_ranks>`.

### Sparse Index Retrieval

The command requires an index and a queries tsv file as input.
```shell
python primeqa/ir/run_ir.py \
      --do_search \
      --engine_type BM25 \
      --index_path <index-dir> \  
      --queries_path  <query_file> \
      --nhits <num-hits> \
      --use_bm25 \
      --k1 <bm25-score-k1> \
      --b <bm25-score-b> \
      --threads  <num-processing-threads> \
      --output_dir <output-dir>
```
The resulting .tsv file, containing query IDs, document IDs, ranks, and scores is stored in `<output-dir>`.

This table shows the sample lines from the search results output file:

| query_id | passage_id | rank | score |
|----|-------|-------|-------|
7606160988275694755|  532|     1|  8.82699966430664|
7606160988275694755|  309305|  2|  8.041299819946289|
7606160988275694755|  65986|   3|  7.9517998695373535|
7606160988275694755|  529090|  4|  7.807199954986572|
