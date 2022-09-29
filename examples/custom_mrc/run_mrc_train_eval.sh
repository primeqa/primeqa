#!/usr/bin/env bash                                                                                                                                                               
set -xeo pipefail

python primeqa/mrc/run_mrc.py \
--model_name_or_path xlm-roberta-large \
--train_file examples/mrc/custom_data/examples_train_squad.jsonl \
--eval_file examples/mrc/custom_data/examples_eval_squad.jsonl \
--preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
--postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
--eval_metrics SQUAD \
--output_dir examples/mrc/output \
--fp16 \
--learning_rate "4e-5" --do_train --do_eval --per_device_train_batch_size 32 --per_device_eval_batch_size 128 \
--gradient_accumulation_steps 2 --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
--num_train_epochs 1 \
--overwrite_output_dir --evaluation_strategy no --overwrite_cache
