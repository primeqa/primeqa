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
python ./script/convert_xorqa_jsonl_to_tsv.py --queries_jsonl_file ./data/xor_dev_retrieve_eng_span_v1.jsonl  --output_file ./data/xortydi_dev.tsv
```
## Model Training, Indexing, Retrieval and Scoring

Reproducing the Dr. DECR XOR-TyDi leaderboard result consists of the following steps:

**Step 1: Fine tuning a ColBERT model using XLM-RoBERTa (XLMR) representation to obtain the teacher model and the student starting-point model, starting from an out-of-box XLMR model**

**Step 2: Two-stage knowledge distillation to train the final (student) model**

**Step 3: Indexing the corpus using the trained student model**

**Step 4: Retrieving the relevent passages using the index**

**Step 5: Reranking the top retrieved passages**

**Step 6: Relevance scoring**

## Step 1: Fine tuning the student and teacher models

There are two rounds of fine tuning involved in this step.
1. Fine tuning using training data containing English queries and passages (NQ) to train the teacher model (the `XLMR -> NQ` step below)
2. Starting from the model from Step 1., fine tuning using training data containing non-English queries and English passages (XOR) to train the student starting-point model (the `NQ -> XOR` step below)

### XLMR -> NQ

```
LR=1.5e-6 python -m colbert.train \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 192 \
    --accum 6 \
    --maxsteps 91287 \
    --triples ./data/ColBERT.C3_3_20_biased200_triples_text.tsv \
    --root ./results/NQ/ \
    --experiment NQ \
    --similarity l2 \
    --run ${LR}_1 \
    --model_type xlm-roberta-base \
    --lr ${LR} \
> ./results/NQ_out.log 
```
The result will be generated in:
```
results/NQ/NQ/train.py/1.5e-6_1/checkpoints/
```

### XLMR -> NQ -> XOR

```
LR=6e-6 python -m colbert.train \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 192 \
    --accum 6 \
    --lr ${LR} \
    --maxsteps 74704 \
    --triples ./data/XOR/data/XOR/xorqa_triples_3poss_100neg_5ep_randTrue.tsv \
    --root ./results/XOR/ \
    --experiment XOR \
    --similarity l2 \
    --run ${LR}_1 \
    --model_type xlm-roberta-base \
    --checkpoint ./results/NQ/NQ/train.py/1.5e-6_1/checkpoints/colbert-LAST.dnn \
> ./results/XOR_out.log 
```

The result will be generated in:
```
results/XOR/XOR/train.py/6e-6_1/checkpoints/
```

In the following Knowledge Distillation (KD) steps, the model resulting from the `XLMR->NQ` training will be used as the teacher model, the model resulting from the `NQ -> XOR` training will be used as the student starting point. 

## Step 2: Two-stage Knowledge Distillation (KD)

### KD with Parallel Corpus Data
```
LR=4.8e-5; QWEIGHT=0.5; python -m colbert.train \
    --amp \
    --doc_maxlen 180 \
    --teacher_doc_maxlen 180 \
    --mask-punctuation \
    --bsize 192 \
    --accum 6 \
    --maxsteps 84897 \
    --lr ${LR} \
    --query_weight ${QWEIGHT} \
    --triples ./data/en-7lan_2ep_triple.other.clean \
    --teacher_triples ./data/en-7lan_2ep_triple.en.clean \
	--root ./results/KD_PC \
    --experiment PC \
    --similarity l2 \
    --run ${LR}_${QWEIGHT}_1 \
    --distill_query_passage_separately True \
    --loss_function MSE \
    --model_type xlm-roberta-base \
    --teacher_model_type xlm-roberta-base \
	--checkpoint ./results/XOR/XOR/train.py/6e-6_1/checkpoints/colbert-LAST.dnn \
	--teacher_checkpoint ./results/NQ/NQ/train.py/1.5e-6_1/checkpoints/colbert-LAST.dnn \
> ./results/KD_PC_out.log ;
```

The result will be generated in:
```
results/KD_PC/PC/train.py/4.8e-5_0.5_1/checkpoints
```

### KD with XOR Data
```
LR=6e-6; python -m colbert.train \
    --amp \
    --doc_maxlen 180 \
    --teacher_doc_maxlen 180 \
    --mask-punctuation \
    --bsize 192 \
    --accum 6 \
    --lr ${LR} \
    --maxsteps 87618 \
    --student_teacher_temperature 2 \
    --triples ./data/XOR/xorqa_triples_3poss_100neg_5ep_randTrue.tsv \
    --teacher_triples ./data/xorqa_triples_3poss_100neg_en_5ep_randTrue.tsv \
	--root ./results/KD_XOR \
    --experiment XOR \
    --similarity l2 \
    --run ${LR}_1 \
    --model_type xlm-roberta-base \
    --teacher_model_type xlm-roberta-base \
    --checkpoint results/KD_PC/PC/train.py/4.8e-5_0.5_1/checkpoints/colbert-LAST.dnn \
	--teacher_checkpoint ./results/NQ/NQ/train.py/1.5e-6_1/checkpoints/colbert-LAST.dnn \
> ./results/KD_XOR_out.log ;
```

The result will be generated in:
```
results/KD_XOR/train.py/6e-6_1/checkpoints
```

## Step 3: Indexing
```
OUTPUT_DIR="./results/post_training/"
CP_PATH="./results/KD_XOR/XOR/train.py/6e-6_1/checkpoints/colbert-LAST.dnn"
mkdir -pv ${OUTPUT_DIR}
CHECKPOINT=colbert-LAST; python -m colbert.index \
    --amp \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 256 \
    --checkpoint ${CP_PATH} \
    --collection ./data/psgs_w100.tsv \
    --index_root ${OUTPUT_DIR} \
    --index_name ${CHECKPOINT}_index \
    --root ${OUTPUT_DIR} \
    --experiment ${CHECKPOINT} \
    --model_type xlm-roberta-base \
    --similarity l2 \
> ${OUTPUT_DIR}/${CHECKPOINT}_index.log ;

CHECKPOINT=colbert-LAST python -m colbert.index_faiss \
    --root ${OUTPUT_DIR}/index_faiss \
    --index_root ${OUTPUT_DIR} \
    --index_name ${CHECKPOINT}_index \
    --partitions 16381 \
    --sample 0.1 \
    --experiment ${CHECKPOINT}_faiss \
> ${OUTPUT_DIR}/${CHECKPOINT}_FAISS.log \
```

The result will be generated in:
```
results/post_training/colbert-LAST_index
```

## Step 4: Retrieval
```
OUTPUT_DIR="../results/post_training/"
CP_PATH="../results/KD_XOR/train.py/6e-6_1/checkpoints/colbert-LAST.dnn"
CHECKPOINT=colbert-LAST python -m colbert.retrieve \
    --doc_maxlen 180 \
    --mask-punctuation \
    --bsize 64 \
    --batch \
    --retrieve_only \
    --queries ./data/xortydi_dev.tsv \
    --index_root ${OUTPUT_DIR} \
    --index_name ${CHECKPOINT}_index \
    --checkpoint ${CP_PATH} \
    --partitions 16381 \
    --faiss_depth 1024 \
    --compression_level -1 \
    --nprobe 32 \
    --similarity l2 \
    --model_type xlm-roberta-base \
    --root ${OUTPUT_DIR} \
    --experiment ${CHECKPOINT}_retrieve \
> ${OUTPUT_DIR}/${CHECKPOINT}_retrieve.log  \
```

The result will be generated in:
```
results/post_training/colbert-LAST_retrieve
```

## Step 5: Reranking
```
OUTPUT_DIR="../results/post_training/"
CP_PATH="../results/KD_XOR/train.py/6e-6_1/checkpoints/colbert-LAST.dnn"
RETRIEVED_FN=\` ls ${OUTPUT_DIR}/${CHECKPOINT}_retrieve/${CHECKPOINT}_retrieve/retrieve.py/*/unordered.tsv \` CHECKPOINT=colbert-LAST python -m colbert.rerank \
    --doc_maxlen 180 \
    --mask-punctuation \
    --similarity l2 \
    --bsize 64 \
    --batch \
    --queries ././data/xortydi_dev.tsv \
    --index_root ${OUTPUT_DIR} \
    --index_name ${CHECKPOINT}_index \
    --checkpoint ${CP_PATH} \
    --model_type xlm-roberta-base \
    --log-scores \
    --partitions 16381 \
    --compression_level -1 \
    --topk \${RETRIEVED_FN} \
    --root ${OUTPUT_DIR} \
    --experiment ${CHECKPOINT}_rerank \
> ${OUTPUT_DIR}/${CHECKPOINT}_rerank.log \
```

The result will be generated in:
```
results/post_training/colbert-LAST_rerank
```

## Step 6: Relavance Scoring

To obtain the relevance scores on the XOR-TyDi development set, run:

```
python ./script/convert_colbert_results_to_xor.py -c ./data/psgs_w100.tsv -q ./data/xor_dev_retrieve_eng_span_v1.jsonl -p ./ranking.tsv -o ./ranking_xortydi_format.json
python ./script/eval_xor_retrieve.py  --data_file ./data/xor_dev_retrieve_eng_span_v1.jsonl --pred_file ./ranking_xortydi_format.json > xorqa.metrics
```

The output in `xorqa.metrics` contains records such as:

```
Evaluating R@2kt
performance on te (238 examples)
79.41176470588235
performance on bn (304 examples)
77.96052631578947
performance on fi (314 examples)
65.92356687898089
performance on ja (241 examples)
63.07053941908713
performance on ko (285 examples)
60.35087719298245
performance on ru (237 examples)
60.75949367088608
performance on ar (309 examples)
65.69579288025889
Final macro averaged score: 
67.59608015198104
Evaluating R@5kt
performance on te (238 examples)
83.19327731092437
performance on bn (304 examples)
82.89473684210526
performance on fi (314 examples)
72.61146496815286
performance on ja (241 examples)
67.63485477178423
performance on ko (285 examples)
68.0701754385965
performance on ru (237 examples)
68.35443037974683
performance on ar (309 examples)
73.13915857605178
Final macro averaged score: 
73.699728326
