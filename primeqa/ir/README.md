<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.ir

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# Information Retrieval (IR)

Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

PrimeQA provides both dense and sparse IR components. 
- There are two **Dense IR** engines supported: ColBERT and Direct Passage Retrieval (DPR).
- **Sparse IR** is a Pyserini-based IR Engine enabling BM25 ranking using bag of words representation.

This README shows how to run the basic model training, data indexing, and retrieval steps using the `run_ir.py` script. 

Sample data files are [here](https://github.com/primeqa/primeqa/tree/main/tests/resources/ir_dense), their formats are shown in the Jupyter notebooks [here](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir.ipynb) and [here](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir_student_teacher.ipynb). 

The steps involved in training a model using the DR.DECR (Dense Retrieval with Distillation-Enhanced Cross-Lingual Representation) Student/Teacher pipeline, as desribed in [Learning Cross-Lingual IR from an English Retriever](https://arxiv.org/abs/2112.08185), are outlined in the [Jupyter notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir_student_teacher.ipynb).

The steps involved in training a model using the DPR-based engine are described in this [Jupyter notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/dense/dense_ir_dpr.ipynb)

The [Jupyter notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/ir/sparse/bm25_retrieval.ipynb) shows how to use the Sparse retriever API.


## Model Training

**Dense IR** requires training a model. The following examples show model training using the `run_ir.py` script.

### Model Training With ColBERT Engine

The script uses training data in a _.tsv_ (tabulator character separated) file, containing training examples in the form of _[query, positive_passage, negative_passage]_ triples. An example of a training data file is [here](https://github.com/primeqa/primeqa/blob/main/tests/resources/ir_dense/xorqa.train_ir_negs_5_poss_1_001pct_at_0pct.tsv).
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
    --doc_maxlen <maximum_number_of_document_tokens> \
    --bsize <training_batch_size> \
    --accum <number_of_gradient_accumulation_steps> \
    --maxsteps <number_of_training_steps> \
    --save_steps <number_of_training_steps_between_saving_checkpoins> \
    --lr <learnig_rate> \
    --model_type xlm-roberta-base \
    --triples <training_data> \
    --root <experiments_root_directory> \
    --experiment <experiment_label>
```

The trained model is stored in `<experiments_root_directory>/<experiment_label>/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn`, with intermediate model files in the same `checkpoints` directory.

### Model Training With DPR Engine

As with the ColBERT engine, the example uses training data in a _.tsv_ (tabulator character separated) file, containing training examples in the form of _[query, positive_passage, negative_passage]_ triples. An example of a training data file with English queries is  [here](https://github.com/primeqa/primeqa/blob/main/tests/resources/ir_dense/xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_en.tsv).

This table shows two lines from the file, with the positive and negative passages truncated:

| query | positive_passage | negative_passage                                                           |
|-------|-------|----------------------------------------------------------------------------|
| Who maintained the throne for the longest time in China? | "Kangxi Emperor The Kangxi Emperors reign of 61 years ... | Chiddy Bang new songs from the duo and in November 2009 debuted...         |
| Who maintained the throne for the longest time in China? | Kangxi Emperor The Kangxi Emperors reign of 61 years ... | Emperor Zhi Yao. The Bamboo Annals says that when Emperor Zhuanxu died ... |

```shell
python primeqa/ir/run_ir.py \
    --do_train \
    --engine_type DPR \
    --train_dir <training_file_or_directory> \
    --output_dir <output_directory> \
    --num_train_epochs <number_of_training_epochs \
    --full_train_batch_size <training_batch_size> \
    --training_data_type text_triples
```

The `--train_dir` contains either a name of a single file, or a directory. If we specify a directory, the script runs training using all files in the directory with the filename extension matching the expected file type (e.g. _.tsv_).

The trained models are stored in `<output_directory>/qry_encoder` and `<output_directory>/ctx_encoder`.

#### Additional Training File Formats for Model Training With DPR Engine

The engine supports the following data formats:

| _-training_data_type_ value | filename extension(s) | description                                                                                                                                                                                                                                                                                                                             |
|-----------------------------|------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| dpr                         | .json, .json.gz | JSON file as in https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-train.json.gz                                                                                                                                                                                                                                            |
| text_triples                | .tsv | [query, positive_passage, negative_passage] triples                                                                                                                                                                                                                                                                                     |
| text_triples_with_title     | .tsv | [query, positive_passage, negative_passage] triples, with the passage fiels containing title and text divided be a vertical bar character                                                                                                                                                                                               |
| num_triples                 | .tsv | [query, positive_passage, negative_passage] triples stored as numerical IDs. <br/>The _.tsv_ file containing the text of queries in the form of [ID, text] is specified in the `--queries` argument. <br/>The _.tsv_ file containing the text of document in the form of [ID, text, title] is specified in the `--collection` argument. |                                                    
| kgi_jsonl                   | .jsonl* | JSONL file containing training examples, as described in https://github.com/IBM/kgi-slot-filling                                                                                                                                                                                                                                        |
 

## Indexing

The following are examples of how to index a collection (set of documents or passages to be searched) using the `run_ir.py` script.

### Corpus Format
The script reads the collection data from a _.tsv_ file, containing collection records in the form of _[ID, text, title]_ triples. The first line of the file contains a header record.
An example of a collection file is [here](https://github.com/primeqa/primeqa/blob/main/tests/resources/ir_dense/xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv).

This table shows the three lines from the file, with _text_ fields truncated:

| id | text | title |
|----|-------|-------|
| 1 | "The Kangxi Emperor's reign of 61 years ... | Kangxi Emperor |
| 2 | Yao. The Bamboo Annals says that when Emperor Zhuanxu died ... | Emperor Zhi |

### Dense Index With ColBERT Engine
Using a model trained as described [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir#model-training-with-colbert-engine), the following command builds the ColBERT index.

```shell
python primeqa/ir/run_ir.py \
    --do_index \
    --engine_type ColBERT \
    --doc_maxlen <maximum_number_of_document_tokens> \
    --bsize <indexing_batch_size> \
    --checkpoint <model_checkpoint> \
    --collection <document_collection> \
    --root <experiments_root_directory> \
    --experiment <experiment_label> \
    --index_name <index_label> \
    --compression_level <number_of_bits_in_residual_representations>
```

The index is stored in `<experiments_root_directory>/<experiment_label>/indexes/<index_label>` directory.

### Dense Index With DPR
Using a model trained as described [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir#model-training-with-dpr-engine), the following command builds the DPR index.

```shell
python primeqa/ir/run_ir.py \
    --engine_type DPR \
    --do_index \
    --dpr_ctx_encoder_path <context_encoder_model> \
    --embed <part_number>of<parts_total> \
    --sharded_index \
    --corpus <document_collection>  \
    --output_dir <output_directory> \
    --batch_size <indexing_batch_size> \
```
Indexing can be parallelized using the `--embed` argument. To accomplish that, we specify the same `parts_total` value (e.g. 16) for all the parallel indexing commands, and specify the `part_number` values (from 1 to `parts_total`) used in the individual commands, e.g. `1of16`, `2of16` to `16of16`.

The resulting index is stored in `output_directory`. 

### Sparse Index With Pyserini

The following command builds an index for BM25 retrieval.

```shell
python primeqa/ir/run_ir.py \
    --do_index \
    --engine_type BM25 \
    --collection <document_collection> \
    --index_path <index_dir>
    --threads <num_threads>
```

## Retrieval

The script uses queries (questions) in a _.tsv_ file in the form of _[ID, text]_ records.
An example of a queries data file is [here](https://github.com/primeqa/primeqa/blob/main/tests/resources/ir_dense/xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv).

### Dense Index Retrieval With Colbert Engine
The command uses a model and index as created in the previous training and indexing steps, described  [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir#model-training-with-colbert-engine) and [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir#dense-index-with-colbert-engine).


```shell
python primeqa/ir/run_ir.py \
    --do_search \
    --engine_type ColBERT \
    --doc_maxlen <maximum_number_of_document_tokens> \
    --bsize <search_batch_size> \
    --queries <query_file> \
    --model_name_or_path <model_filename_or_directory> \
    --index_location <directory_containing_index_files> \
    --top_k <number_of_items_per_query_retrieved> \
    --output_dir <output_directory>
```

The resulting .tsv file, containing query IDs, document IDs, ranks, and scores is stored in `<output_directory>`, in a file named `ranked_passages.tsv`.

#### PLAID hyperparameters

The hyperparameters `ncells`, `centroid_score_threshold`, and `ndocs` can optionally be used to trade off between retrieval accuracy and speed. If these are not set explicitly they instead default to pre-configured values based on `k` as shown in the following table:

|       `k`       | `ncells` | `centroid_score_threshold` |       `ndocs`      |
|:---------------:|:--------:|:--------------------------:|:------------------:|
|    `k` <= 10    |     1    |             0.5            |         256        |
| 10 < `k` <= 100 |     2    |            0.45            |        1024        |
|    100 < `k`    |     4    |             0.4            | max(`k` * 4, 4096) |

See the [PLAID paper](https://arxiv.org/abs/2205.09707) for more details.

Note that the previous ColBERTv2 hyperparameters `nprobe` and `ncandidates` are now deprecated.

### Dense Index Retrieval With DPR Engine
The command uses a model and index as created in the previous training and indexing steps, described  [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir#model-training-with-dpr-engine) and [here](https://github.com/primeqa/primeqa/tree/main/primeqa/ir#dense-index-with-dpr-engine).

```shell
python primeqa/ir/run_ir.py \
    --do_search \
    --engine_type DPR \
    --queries <query_file> \
    --model_name_or_path <query_encoder_model> \
    --bsize <search_batch_size> \
    --index_location <directory_containing_index_files> \
    --top_k <number_of_items_per_query_retrieved> \
    --output_dir <output_directory>
```

The engine uses a default tokenizer (currently `facebook/dpr-ctx_encoder-multiset-base`).  If needed, the tokenizer may be specified using the `--qry_tokenizer_path` argument.

The resulting .tsv file, containing query IDs, document IDs, ranks, and scores is stored in `<output_directory>`, in a file named `ranked_passages.tsv`.

### Sparse Index Retrieval

The command requires an index and a queries tsv file as input.

```shell
python primeqa/ir/run_ir.py \
      --do_search \
      --engine_type BM25 \
      --index_location <index-dir> \
      --queries  <query_file> \ 
      --output_dir <output-dir>
```
The resulting .tsv file, containing query IDs, document IDs, ranks, and scores is stored in `<output-dir>` in 'ranked_passages.tsv' file.

This table shows the sample lines from the search results tsv file:

| query_id            | passage_id | rank | score |
|---------------------|-------|-------|-------|
| 7606160988275694755 |  532|     1|  8.82699966430664|
| 7606160988275694755 |  309305|  2|  8.041299819946289|
| 7606160988275694755 |  65986|   3|  7.9517998695373535|
| 7606160988275694755 |  529090|  4|  7.807199954986572|


## Scoring
The scoring steps depend on the task and metric used.
The following describes the steps to evaluate retrieval on the [XOR-TyDi task](https://nlp.cs.washington.edu/xorqa/)

### Obtain XORTyDI GroundTruth data and Eval scripts
1. Clone the XORTyDI repo here: https://github.com/AkariAsai/XORQA
2. Download ground truth data file here: https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_full_v1_1.jsonl


### Convert for evaluation
Convert the search results obtained from the retrieval step into the format required by the evaluation script.  This is done by running the conversion script:

```shell
    python primeqa/ir/scripts/xortydi/convert_colbert_results_to_xor.py \
    -c <document_collection> \
    -q <ground_truth_data>
    -p <scores_and_ranks> \
    -o <ranking_xortydi_format.json>
```

### Run Evaluation Script

Run:
   ```shell
   python <path-to-xorqa-repo>/evals/eval_xor_retrieve.py --data_file <ground_truth_data> --pred_file <ranking_xortydi_format.json>
   ```

### Sample Evaluation Results
The following is an example of evaluation script output obtained by running Sparse Retrival using the following collection and query set:
1. Index the DPR corpus of English Wikpedia (December 20, 2018 dump) split 100 word passages
   `wget https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz`
2. English MT translation of XORTyDI DEV queries from [here](https://drive.google.com/file/d/1JzlNDijDZmDlT42ABVJK53gwk7_mKHGt/view?usp=sharing).

```
Evaluating R@2kt
performance on te (238 examples)
53.78151260504202
performance on bn (304 examples)
62.82894736842105
performance on fi (314 examples)
43.31210191082803
performance on ja (241 examples)
42.32365145228216
performance on ko (285 examples)
43.15789473684211
performance on ru (237 examples)
54.85232067510548
performance on ar (309 examples)
49.19093851132686
Final macro averaged score: 
49.921052465692526
Evaluating R@5kt
performance on te (238 examples)
63.02521008403361
performance on bn (304 examples)
70.06578947368422
performance on fi (314 examples)
51.910828025477706
performance on ja (241 examples)
53.52697095435685
performance on ko (285 examples)
53.333333333333336
performance on ru (237 examples)
61.60337552742617
performance on ar (309 examples)
57.60517799352751
Final macro averaged score: 
58.72438362740563
```


