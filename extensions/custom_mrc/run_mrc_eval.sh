#!/usr/bin/env bash                                                                                                                                                               
set -xeo pipefail

python primeqa/mrc/run_mrc.py \
--model_name_or_path PrimeQA/squad-v1-roberta-large \
--eval_file examples/mrc/custom_data/examples_eval_squad.jsonl \
--preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
--postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
--eval_metrics SQUAD \
--output_dir examples/mrc/output \
--fp16 \
--do_eval --per_device_eval_batch_size 128 \
--overwrite_output_dir --overwrite_cache
