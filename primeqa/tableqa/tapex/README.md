# Table Question Answering using TAPEX

TAPEX is a table pre-training approach for table-related tasks. By learning a neural SQL executor over a synthetic corpus based on generative language models (e.g., BART), it achieves state-of-the-art performance on several table question answering and table fact verification benchmarks. More details can be found in the original paper [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/pdf/2107.07653.pdf).

PrimeQA supports TAPEX based training and eval over two major table question answering datasets like wikisql, wikitablequestions
Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

## TAPEX based TableQA models

### Train/Eval using TAPEX based Table Question Answering model in PrimeQA on wikitablequestions dataset
```
python run_tapex.py \
  --do_train \
  --do_eval \
  --dataset_name wikitablequestions \
  --output_dir tapex_wtq\
  --model_name_or_path microsoft/tapex-large \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --logging_steps 10 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000 

```
The trained model yields the following results on wikitablequestions dev set:
```
***** eval metrics *****
  eval_denotation_accuracy =     0.5793
  eval_samples             =       2831
  eval_samples_per_second  =     10.443
  eval_steps_per_second    =      2.612
```
### Train/Eval using TAPEX based Table Question Answering model in PrimeQA on wikisql dataset

```
python run_tapex.py \
  --do_train \
  --do_eval \
  --dataset_name wikisql \
  --output_dir tapex_base_wikiql\
  --model_name_or_path microsoft/tapex-base \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 4 \
  --learning_rate 3e-5 \
  --logging_steps 10 \
  --eval_steps 1000 \
  --save_steps 1000 \
  --warmup_steps 1000 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --max_steps 20000 

```
The trained model yields the following results on wikitablequestions dev set:
```
***** eval metrics *****
  epoch                    =      49.99
  eval_denotation_accuracy =      0.598
  eval_loss                =     2.4315
  eval_runtime             = 0:08:46.28
  eval_samples             =       2831
  eval_samples_per_second  =      5.379
  eval_steps_per_second    =      0.336

```

### Inference

```
The trained model yields the following results on wikisql dev set:
```
***** eval metrics *****
  eval_denotation_accuracy =     0.8800
  eval_samples             =       8421
  eval_samples_per_second  =     18.986
  eval_steps_per_second    =      4.748
```

TAPEX can also be used from PrimeQA built-in-class TapexReader to do train/eval/inference with minimal line of codes. See example [notebooks](https://github.com/primeqa/primeqa/tree/tapex_integration/notebooks/tableqa) for the same. 
