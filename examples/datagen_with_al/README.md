# Training with Active Learning

TODO: add description, reference paper

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

The scripts `run.py` and `run_mrc.py` as used below can be modified using the [TrainingArguments](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments) from the transformers libray.

## AL for Data Generation

### Prerequisites

AL usually works best if a model trained on supervised data exists.
In our setting we start with models fine-tuned on SQuAD.
Therefore let's train the data generation model on SQuAD first:

```bash
python run.py \
    --do_train \
    --model_name facebook/bart-large \
    --output_dir models/qg/bart-large_squad \
    --train_dataset st-squad:train \
    --eval_dataset st-squad:validation \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --save_total_limit 1 \
    --metric_for_best_model loss \
    --num_train_epochs 5 \
    --per_device_eval_batch_size 5 \
    --learning_rate 3e-5 \
    --num_worker 5 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 1000 \
    --save_steps 1000
```

### Apply AL

Next, we fine-tune using AL for sample selection.

```bash
python run.py \
    --do_al \
    --model_name models/qg/bart-large_squad/checkpoint-30 \ # make sure to pick the best checkpoint from previous training here, i.e. the one with the lowest number of steps
    --output_dir models/qg/bart-large-squad_al-bioasq-dsp-4x50 \
    --dataset_name bioasq \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 24 \
    --save_total_limit 1 \
    --metric_for_best_model loss \
    --max_steps 500 \ # or num_train_epochs but dataset size is really small
    --per_device_eval_batch_size 5 \
    --learning_rate 3e-5 \
    --num_worker 5 \
    --eval_steps 10 \
    --logging_steps 10 \
    --save_steps 10
```

### Generate data

In this step, we will generate question-answer pairs from unlabelled documents given a fine-tuned data generation model, e.g. after applying AL in the previous step:

```bash
python run.py \
    --do_generate \
    --model_name models/qg/bart-large-squad_al-bioasq-dsp-4x50_round-3-of-3/checkpoint-100 \ # make sure to pick the best checkpoint from last iteration of AL training here, i.e. the one with the lowest number of steps
    --output_dir tmp \
    --dataset_name pubmed \
    --per_device_eval_batch_size 5 \
    --num_worker 5 \
    --max_gen_length 300 \
    --gen_output_path generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50
```

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
python run_mrc.py \
    --model bert-base-uncased \
    --output_dir models/rc/squad_eval-bioasq \
    --train_dataset st-squad:train \
    --eval_dataset st-bioasq:validation \
    --fp16 \
    --learning_rate 4e-5 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --save_total_limit 1 \
    --num_train_epochs 3 \
    --overwrite_cache \
    --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
    --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --eval_metrics SQUAD \
    --metric_for_best_model f1 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 1000 \
    --save_steps 1000

# further fine-tune SQuAD-trained RC model on annotated data from BioASQ
python run_mrc.py \
    --model models/rc/squad_eval-bioasq/checkpoint-8000 \ # select best checkpoint from previous training, i.e. checkpoint with least steps
    --output_dir models/rc/squad_bioasq-al-dsp-4x50 \
    --train_dataset models/bart-large-squad_bioasq-al-dsp-4x50/ \
    --eval_dataset st-bioasq:validation \
    --fp16 \
    --learning_rate 4e-5 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --save_total_limit 1 \
    --max_steps 200 \
    --overwrite_cache \
    --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
    --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --eval_metrics SQUAD \
    --metric_for_best_model f1 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --logging_steps 10 \
    --save_steps 10
```

This can then be used for filtering:
```bash
python filter_gen_data.py generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50 generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50_rt --filter rt --rt-model models/rc/squad_bioasq-al-dsp-4x50/checkpoint-220 # again, pick best checkpoint
```


### Train RC

Train an RC model using the generated & filtered data as well as annotated data:

```bash
python run_mrc.py \
    --model bert-base-uncased \
    --output_dir models/rc/bioasq-bart-large-squad-al-bioasq-dsp-4x50_bioasq-al-dsp-4x50_bioasq-al-dsp-4x50 \
    --train_dataset generated_data/bioasq_bart-large-squad-al-bioasq-dsp-4x50_lm models/bart-large-squad_al-bioasq-dsp-4x50/al_samples_round_3 \
    --eval_dataset st-bioasq:validation \
    --fp16 \
    --learning_rate 4e-5 \
    --do_train \
    --do_eval \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 2 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --save_total_limit 1 \
    --num_train_epochs 3 \
    --overwrite_cache \
    --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
    --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --eval_metrics SQUAD \
    --metric_for_best_model f1 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --logging_steps 1000 \
    --save_steps 1000
```
