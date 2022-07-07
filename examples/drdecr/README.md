# Training and Running Dr. DECR Models

This file provides instructions on how to train and test the Dr. DECR models (Dense Retrieval with Distillation-Enhanced Cross-Lingual Representation) Student/Teacher pipeline, as desribed in [Learning Cross-Lingual IR from an English Retriever](https://arxiv.org/abs/2112.08185). 

Similar models based on ColBERT V1, were used to obtain the results in Dr. DECR XOR-TyDi leaderboard whitebox (not using external APIs) submission.

## Installation

The following steps require PrimeQA to be [installed](../../README.md#Installation). 

## Data Preparation

There are three datasets used to train Dr. DECR:
* NQ (data containing English queries and passages)
* XOR (data containg English queries and non-English passages)
* Parallel Corpus (data containg parallel English and non-English passages)

More details about the data sets can be found in Appendix A.1 of the [Dr. DECR](https://arxiv.org/abs/2112.08185) paper.

To create **NQ** training data, run:
```
source ./script/create_NQ.sh
```
The output is in:
```
./data/ColBERT.C3_3_20_biased200_triples_text.tsv
./data/psgs_w100.tsv
```

To create **XOR** training data:

first create a `./data/XOR` directory and download:
```
https://drive.google.com/file/d/1tB0ehH0Q_V7gEgc2ZoxSlXpHTzUONNNW/view
```
into the directory, then run:
```
source ./script/create_XOR.sh
```
The output files are: 
```
./data/xortydi_ir_negs_poss.json
./data/xorqa_triples_3poss_100neg_5ep_randTrue.tsv (17125570 lines)
./data/xorqa_triples_3poss_100neg_en_5ep_randTrue.tsv (17125570 lines)
```

To generate **Parallel Corpus** training data, run:
```
source ./script/create_PC.sh
```
The output is in:
```
./data/en-7lan_2ep_triple.en.clean
./data/en-7lan_2ep_triple.other.clean
```

NQ and XOR training files use the following TSV format: `{query \t positive passage \t negative passage}`

Parallel Corpus training files use the following TSV format: `{ {English, non-English} passage \t English passage \t English passage}` to be consistent with other training data

To download and pre-process additional data used in indexing and scoring, run:
```
wget --output-document ./data/xor_dev_retrieve_eng_span_v1.jsonl https://nlp.cs.washington.edu/xorqa/XORQA_site/data/xor_dev_retrieve_eng_span_v1.jsonl
python ../../primeqa/ir/scripts/xortydi/convert_xorqa_jsonl_to_tsv.py --queries_jsonl_file ./data/xor_dev_retrieve_eng_span_v1.jsonl  --output_file ./data/xortydi_dev.tsv
```
## Model Training, Indexing, Retrieval and Scoring

Reproducing the Dr. DECR XOR-TyDi leaderboard result consists of the following steps:

**Step 1: Fine tuning a ColBERT model using XLM-RoBERTa (XLMR) representation to obtain the teacher model and the student starting-point model, starting from an out-of-box XLMR model**

**Step 2: Two-stage knowledge distillation to train the final (student) model**

**Step 3: Indexing the corpus using the trained student model**

**Step 4: Retrieving the relevent passages using the index**

**Step 5: Relevance scoring**

## Step 1: Fine tuning the student and teacher models

There are two rounds of fine tuning involved in this step.
1. Fine tuning using training data containing English queries and passages (NQ) to train the teacher model (the `XLMR -> NQ` step below)
2. Starting from the model from Step 1., fine tuning using training data containing non-English queries and English passages (XOR) to train the student starting-point model (the `NQ -> XOR` step below)

### XLMR -> NQ

```
python examples/ir/run_ir.py \
    --do_train \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --lr 1.5e-6 \
    --bsize 192 \
    --accum 6 \
    --maxsteps 91287 \
    --triples ./data/ColBERT.C3_3_20_biased200_triples_text.tsv \
    --root ./results \
    --experiment NQ \
    --similarity l2 \
    --model_type xlm-roberta-base \
> ./results/NQ_out.log 
```
The trained model will be stored in:
`
./results/NQ/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn 
`

### NQ -> XOR

```
python examples/ir/run_ir.py \
    --do_train \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --lr 6e-6 \
    --bsize 192 \
    --accum 6 \
    --maxsteps 74704 \
    --triples ./data/XOR/data/XOR/xorqa_triples_3poss_100neg_5ep_randTrue.tsv \
    --root ./results \
    --experiment XOR \
    --similarity l2 \
    --model_type xlm-roberta-base \
    --checkpoint ./results/NQ/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn \
> ./results/XOR_out.log 
```

The trained model will be stored in:
`
./results/XOR/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn`

In the following Knowledge Distillation (KD) steps, the model resulting from the `XLMR->NQ` training will be used as the teacher model, and the model resulting from the `NQ -> XOR` training will be used as the student starting point. 

## Step 2: Two-stage Knowledge Distillation (KD)

### KD with Parallel Corpus Data
```
python examples/ir/run_ir.py \
    --do_train \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --teacher_doc_maxlen 180 \
    --mask-punctuation \
    --bsize 192 \
    --accum 6 \
    --maxsteps 84897 \
    --lr 4.8e-5 \
    --query_weight 0.5 \
    --triples ./data/en-7lan_2ep_triple.other.clean \
    --teacher_triples ./data/en-7lan_2ep_triple.en.clean \
	--root ./results/KD_PC \
    --experiment PC \
    --similarity l2 \
    --distill_query_passage_separately True \
    --loss_function MSE \
    --model_type xlm-roberta-base \
    --teacher_model_type xlm-roberta-base \
    --checkpoint ./results/XOR/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn \
    --teacher_checkpoint ./results/NQ/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn \
> ./results/KD_PC_out.log ;
```

The trained model will be stored in:
`
results/KD_PC/PC/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn
`

### KD with XOR Data
```
python examples/ir/run_ir.py \
    --do_train \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --teacher_doc_maxlen 180 \
    --mask-punctuation \
    --bsize 192 \
    --accum 6 \
    --lr 6e-6 \
    --maxsteps 87618 \
    --student_teacher_temperature 2 \
    --triples ./data/XOR/xorqa_triples_3poss_100neg_5ep_randTrue.tsv \
    --teacher_triples ./data/xorqa_triples_3poss_100neg_en_5ep_randTrue.tsv \
	--root ./results/KD_XOR \
    --experiment XOR \
    --similarity l2 \
    --model_type xlm-roberta-base \
    --teacher_model_type xlm-roberta-base \
    --checkpoint ./results/KD_PC/PC/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn \
    --teacher_checkpoint ./results/NQ/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn \
> ./results/KD_XOR_out.log ;
```

The trained model will be stored in:
`
results/KD_XOR/XOR/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn
`

## Step 3: Indexing
```
OUTPUT_DIR="./results/post_training/" ; \
mkdir -pv ${OUTPUT_DIR} ; \
CP_PATH="./results/KD_XOR/XOR/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn" ; \
CHECKPOINT=colbert-LAST; 
python examples/ir/run_ir.py \
    --do_index \
    --engine_type ColBERT \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 256 \
    --similarity l2 \
    --checkpoint ${CP_PATH} \
    --collection ./data/psgs_w100.tsv \
    --index_name ${CHECKPOINT}_index \
    --root ${OUTPUT_DIR} \
    --experiment ${CHECKPOINT} \
    --model_type xlm-roberta-base \
    --compression_level 2 \
> ${OUTPUT_DIR}/${CHECKPOINT}_index.log ;
```

The index will be stored in:
`./results/post_training/colbert-LAST/colbert-LAST_index` directory.

## Step 4: Retrieval
```
OUTPUT_DIR="./results/post_training/" ; \
CP_PATH="./results/KD_XOR/XOR/none/<year_month/<day>/<time>/checkpoints/colbert-LAST.dnn" ; \
 CHECKPOINT=colbert-LAST ; \
python examples/ir/run_ir.py \
    --do_search \
    --engine_type ColBERT \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 4 \
    --similarity l2 \
    --retrieve_only \
    --queries ./data/xortydi_dev.tsv \
    --checkpoint ${CP_PATH} \
    --root ${OUTPUT_DIR} \
    --index_name ${CHECKPOINT}_index \
    --experiment ${CHECKPOINT}_retrieve \
    --nprobe 4 \
    --ranks_fn ${OUTPUT_DIR}/colbert-LAST_retrieve.tsv \
> ${OUTPUT_DIR}/${CHECKPOINT}_retrieve.log
```

The resulting .tsv file, containing query IDs, document IDs, ranks, and scores will be stored in:
`
./results/post_training/colbert-LAST_retrieve.tsv
`

## Step 5: Relevance Scoring

To obtain the relevance scores on the XOR-TyDi development set, the scores have to be converted into XOR-TyDi format by running:

```
   python primeqa/ir/scripts/xortydi/convert_colbert_results_to_xor.py \
    -c ./data/psgs_w100.tsv \
    -q ./data/xor_dev_retrieve_eng_span_v1.jsonl \
    -p ./results/post_training/colbert-LAST_retrieve.tsv \
    -o ./results/post_training/colbert-LAST_retrieve_xortydi_format.json
```

Finally, to obtain the XOR-TyDi scores, run:
```
python eval_xor_retrieve.py \
    --data_file ./data/xor_dev_retrieve_eng_span_v1.jsonl \
    --pred_file ./results/post_training/colbert-LAST_retrieve_xortydi_format.json > ./results/post_training/xorqa.metrics
```

The `eval_xor_retrieve.py` script can be downloaded from the XORTyDI repo here: https://github.com/AkariAsai/XORQA

The output in `./results/post_training/xorqa.metrics` will contain records such as:

```
Evaluating R@2kt
performance on te (238 examples)
74.36974789915966
performance on bn (304 examples)
73.35526315789474
performance on fi (314 examples)
61.78343949044586
performance on ja (241 examples)
50.20746887966805
performance on ko (285 examples)
60.0
performance on ru (237 examples)
54.43037974683544
performance on ar (309 examples)
56.310679611650485
Final macro averaged score: 
61.49385411223632
Evaluating R@5kt
performance on te (238 examples)
78.99159663865547
performance on bn (304 examples)
81.25
performance on fi (314 examples)
67.83439490445859
performance on ja (241 examples)
59.33609958506224
performance on ko (285 examples)
67.36842105263158
performance on ru (237 examples)
63.29113924050633
performance on ar (309 examples)
64.07766990291263
Final macro averaged score: 
68.87847447488954
```
