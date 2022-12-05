# Table Question Answering using Tapex and OmniTab
PrimeQA supports training and inference over all major table question answering datasets like wikisql, wikitablequestions etc.
Users can fine-tune a model to do question answering using tapex and Omnitab pre-trained models.
Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).


## TAPEX based TableQA models
### Training Tapex based Table Question Answering model using PrimeQA on wikitablequestions dataset

```
python run_tapex.py \
  --do_train \
  --do_eval \
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

### Inference

```
python run_tapex.py \
  --do_eval \
  --model_name_or_path tapex_wtq/checkpoint-20000 \
  --output_dir eval-trained-tapex-large-wtq \
  --per_device_eval_batch_size 4 \
  --predict_with_generate \
  --num_beams 5

```

## OmniTab base TableQA models

### Training OmniTab Based TableQA Model

```
python run_tapex.py \
  --do_train \
  --do_eval \
  --output_dir omnitab_wtq\
  --model_name_or_path neulab/omnitab-large \
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

### Inference

```
python run_tapex.py \
  --do_eval \
  --model_name_or_path omnitab_wtq/checkpoint-20000 \
  --output_dir eval-trained-omnitab-large-wtq \
  --per_device_eval_batch_size 4 \
  --predict_with_generate \
  --num_beams 5
```