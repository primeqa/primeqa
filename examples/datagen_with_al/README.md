# Training with Active Learning

## Installation

The following steps require PrimeQA to be [installed](../../README.md#Installation).

## Prerequisites

Before we start, I suggest to set the cache directories unless you have enough space in `$HOME/.cache`:
```bash
CACHE_DIR = path/to/dir # set this to a custom directory where you want to have huggingface downloads cached
export TRANSFORMERS_CACHE=$CACHE_DIR
export HF_DATASETS_CACHE=$CACHE_DIR
export HF_MODULES_CACHE=$CACHE_DIR
```

If you use CUDA, make sure to also set `CUDA_VISIBLE_DEVICES` to the appropriate device.
E.g. for GPU 2 (the third one) on the current host:
```bash
export CUDA_VISIBLE_DEVICES=2
```

The script `run.py` as used below can be modified using the [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) from the transformers libray.

## AL for Data Generation

### Prerequisites

AL usually works best if a model trained on supervised data exists.
In our setting we start with models fine-tuned on SQuAD.
Therefore let's train the data generation model on SQuAD first:

```bash
python run.py \
    --train_dataset st-squad:train \
    --eval_dataset st-squad:validation \
    --num_workers 5 \
    --rc_output_dir tmp \
    --qg_do_train \
    --qg_model_name facebook/bart-large \
    --qg_output_dir models/qg/bart-large_squad \
    --qg_per_device_train_batch_size 1 \
    --qg_gradient_accumulation_steps 24 \
    --qg_save_total_limit 1 \
    --qg_metric_for_best_model loss \
    --qg_num_train_epochs 5 \
    --qg_per_device_eval_batch_size 5 \
    --qg_learning_rate 3e-5 \
    --qg_evaluation_strategy steps \
    --qg_eval_steps 1000 \
    --qg_logging_steps 1000 \
    --qg_save_steps 1000
```

### Apply AL

Next, we fine-tune using AL for sample selection.

```bash
python run.py \
    --do_al \
    --train_dataset st-bioasq:train \
    --eval_dataset st-bioasq:validation \
    --num_workers 5 \
    --output_dir models/qg/bart-large-squad_al-bioasq-dsp-4x50 \
    --rc_output_dir tmp \
    --qg_output_dir tmp \
    --qg_model_name models/qg/bart-large_squad/checkpoint-12000 \
    --qg_per_device_train_batch_size 1 \
    --qg_gradient_accumulation_steps 24 \
    --qg_save_total_limit 1 \
    --qg_metric_for_best_model loss \
    --qg_max_steps 500 \
    --qg_per_device_eval_batch_size 5 \
    --qg_learning_rate 3e-5 \
    --qg_eval_steps 10 \
    --qg_logging_steps 10 \
    --qg_save_steps 10
```
> Make sure to pick the best checkpoint from previous training for `--qg_model_name`, i.e. the one with the lowest number of steps.
> Also we're setting `--qg_max_steps` since dataset size is small and `--num_train_epochs` would result in too few training steps (if not set to a huge value)

### Generate data

In this step, we will generate question-answer pairs from unlabelled documents given a fine-tuned data generation model, e.g. after applying AL in the previous step:

```bash
python run.py \
    --do_generate \
    --predict_dataset pubmed-20 \
    --num_workers 5 \
    --rc_output_dir tmp \
    --max_gen_length 300 \
    --gen_output_path generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50 \
    --qg_model_name models/qg/bart-large-squad_al-bioasq-dsp-4x50_round-4-of-4/checkpoint-500 \
    --qg_output_dir tmp \
    --qg_per_device_eval_batch_size 5
```
> Make sure to pick the best checkpoint from last iteration of AL for `--qg_model_name`, i.e. the one with the lowest number of steps.
> 

By default, the generated data is stored as a `datasets:Dataset` to `./generated_data`. This can be changed using `--gen_output_path`.
> Note that you can use sharding to split a large set of documents into several runs by specifying `--shard_size` or `--num_shards`. The shard indices are then computed deterministically and the zero-based shard indices to run can optionally be set using `--shards`. Generated data is then stored in subfolders of the output directory named with the indices of the shards.

### Filter generated data

Since generated data is noisy, we will filter them in a next step.
The script `filter_gen_data.py` does this:
```bash
python filter_gen_data.py <paths to generated data> <output dir> --filter {lm,rt}
```
> Note that in case of sharding applied to the data generation, you will have to give all shards to the script, for example `generated_data/*`.

There are two filtering methods integrated: One based on LM score and the other using an RC model.

#### LM score filtering

For applying filtering using the LM score:
```bash
python filter_gen_data.py generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50 generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50_lm --filter lm
```

#### RTcons filtering

For applying filtering using an RC model we need to train an RC model first.
We suggest fine-tuning a model on SQuAD first followed by fine-tuning on the selected, annotated data:
```bash
# fine-tune RC model on SQuAD
python run.py \
    --train_dataset st-squad:train \
    --eval_dataset st-bioasq:validation \
    --qg_output_dir tmp \
    --rc_model bert-base-uncased \
    --rc_output_dir models/rc/squad_eval-bioasq \
    --rc_fp16 \
    --rc_learning_rate 4e-5 \
    --rc_do_train \
    --rc_do_eval \
    --rc_per_device_train_batch_size 16 \
    --rc_per_device_eval_batch_size 128 \
    --rc_gradient_accumulation_steps 2 \
    --rc_warmup_ratio 0.1 \
    --rc_weight_decay 0.01 \
    --rc_save_total_limit 1 \
    --rc_num_train_epochs 3 \
    --rc_preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
    --rc_postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --rc_eval_metrics SQUAD \
    --rc_metric_for_best_model f1 \
    --rc_evaluation_strategy steps \
    --rc_eval_steps 1000 \
    --rc_logging_steps 1000 \
    --rc_save_steps 1000
```
```bash
# further fine-tune SQuAD-trained RC model on annotated data from BioASQ
# select best checkpoint from previous training, i.e. checkpoint with least steps
python run.py \
    --train_dataset models/qg/bart-large-squad_al-bioasq-dsp-4x50/al_samples_round_3 \
    --eval_dataset st-bioasq:validation \
    --qg_output_dir tmp \
    --rc_model models/rc/squad_eval-bioasq/checkpoint-8000 \
    --rc_output_dir models/rc/squad_ft-bioasq-al-dsp-4x50 \
    --rc_fp16 \
    --rc_learning_rate 4e-5 \
    --rc_do_train \
    --rc_do_eval \
    --rc_per_device_train_batch_size 16 \
    --rc_per_device_eval_batch_size 128 \
    --rc_gradient_accumulation_steps 2 \
    --rc_warmup_ratio 0.1 \
    --rc_weight_decay 0.01 \
    --rc_save_total_limit 1 \
    --rc_max_steps 500 \
    --rc_preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
    --rc_postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --rc_eval_metrics SQUAD \
    --rc_metric_for_best_model f1 \
    --rc_evaluation_strategy steps \
    --rc_eval_steps 10 \
    --rc_logging_steps 10 \
    --rc_save_steps 10
```

This can then be used for filtering:
```bash
python filter_gen_data.py generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50 generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50_rt --filter rt --rt_model models/rc/squad_ft-bioasq-al-dsp-4x50/checkpoint-220 --num_workers 5 # again, pick best checkpoint
```


### Train RC

Train an RC model using the generated & filtered data as well as annotated data:

```bash
python run.py \
    --train_dataset generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50_lm models/qg/bart-large-squad_al-bioasq-dsp-4x50/al_samples_round_3 \
    --eval_dataset st-bioasq:validation \
    --qg_output_dir tmp \
    --rc_model bert-base-uncased \
    --rc_output_dir models/rc/bioasq-bart-large-squad-al-bioasq-dsp-4x50_bioasq-al-dsp-4x50_bioasq-al-dsp-4x50 \
    --rc_fp16 \
    --rc_learning_rate 4e-5 \
    --rc_do_train \
    --rc_do_eval \
    --rc_per_device_train_batch_size 16 \
    --rc_per_device_eval_batch_size 128 \
    --rc_gradient_accumulation_steps 2 \
    --rc_warmup_ratio 0.1 \
    --rc_weight_decay 0.01 \
    --rc_save_total_limit 1 \
    --rc_num_train_epochs 3 \
    --rc_preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
    --rc_postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --rc_eval_metrics SQUAD \
    --rc_metric_for_best_model f1 \
    --rc_evaluation_strategy steps \
    --rc_eval_steps 1000 \
    --rc_logging_steps 1000 \
    --rc_save_steps 1000
```
