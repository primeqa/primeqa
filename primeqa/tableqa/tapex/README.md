# Table Question Answering using Tapex
PrimeQA supports Tapex based training and eval over two major table question answering datasets like wikisql, wikitablequestions
Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).


## TAPEX based TableQA models
### Train/Eval using Tapex based Table Question Answering model in PrimeQA on wikitablequestions dataset

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
### Train/Eval using Tapex based Table Question Answering model in PrimeQA on wikisql dataset
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

Tapex can also be used from PrimeQA built-in-class TapexReader to do train/eval/inference with minimal line of codes. See example [notebooks](https://github.com/primeqa/primeqa/tree/tapex_integration/notebooks/tableqa) for the same. 
