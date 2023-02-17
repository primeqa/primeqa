# Table Question Answering using TAPEX

TAPEX is a table pre-training approach for table-related tasks. By learning a neural SQL executor over a synthetic corpus based on generative language models (e.g., BART), it achieves state-of-the-art performance on several table question answering and table fact verification benchmarks. More details can be found in the original paper [TAPEX: Table Pre-training via Learning a Neural SQL Executor](https://arxiv.org/pdf/2107.07653.pdf).

PrimeQA supports TAPEX based training and eval over two major table question answering datasets like wikisql, WikiTableQuestions.
Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

## TAPEX based TableQA models

### Train/Eval using TAPEX based Table Question Answering model in PrimeQA on WikiTableQuestions dataset
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
The trained model yields the following results on WikiTableQuestions dev set:
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
The trained model yields the following results on wikisql dev set:
```
***** eval metrics *****
  eval_denotation_accuracy =     0.8800
  eval_samples             =       8421
  eval_samples_per_second  =     18.986
  eval_steps_per_second    =      4.748
```


## Omnitab based TableQA models

OmniTab is an omnivorous pretraining approach that consumes natural data to endow models with the ability to understand and align natural language with tables, and synthetic questions to train models to perform reasoning. More details can be found in the original paper [OmniTab: Pretraining with Natural and Synthetic Data for Few-shot Table-based Question Answering](https://arxiv.org/pdf/2207.03637.pdf).

PrimeQA supports OmniTab based tableqa model training and inference over WikiTableQuestions dataset.
Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

### Train/Eval using OmniTab based Table Question Answering model in PrimeQA on WikiTableQuestions dataset
```
python run_tapex.py \
  --do_train \
  --do_eval \
  --dataset_name wikitablequestions \
  --output_dir omnitab_wtq \
  --max_source_length 1024 \
  --max_target_length 128 \
  --model_name_or_path neulab/omnitab-large \
  --overwrite_output_dir \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-5 \
  --logging_steps 10 \
  --eval_steps 1000 \
  --val_max_target_length 128 \
  --save_steps 1000 \
  --warmup_steps 1000 \
  --warmup_ratio 0.1 \
  --evaluation_strategy steps \
  --predict_with_generate \
  --num_beams 5 \
  --weight_decay 1e-2 \
  --label_smoothing_factor 0.1 \
  --generation_max_length 128 \
  --max_steps 20000 

```
The trained model yields the following results on WikiTableQuestions dev set:
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

TAPEX and OmniTab can also be used from PrimeQA built-in-class TapexReader to do train/eval/inference with minimal line of codes. See example [notebooks](https://github.com/primeqa/primeqa/notebooks/tableqa) for the same. 
