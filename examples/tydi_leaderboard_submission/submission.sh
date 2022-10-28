#!/usr/bin/env bash
#
# submission.sh: The script that runs your system.
#
# Usage: submission.sh <input_path> <output_path>
#   input_path: File pattern (e.g. <input dir>/tydiqa-test-??.jsonl.gz).
#   output_path: Path to JSONL file containing predictions (e.g. predictions.jsonl).
#
# Sample usage:
#   submission.sh input_path output_path

set -xeo pipefail

set -u  # This is after conda setup as conda setup has unbound vars which trigger it

INPUT_PATH=$1
OUTPUT_PATH=$2

# TODO add back --fp16

MRC_MODEL='/model/models/mrc'
BOOLEAN_CONFIG='/model/examples/tydi_leaderboard_submission/tydi_boolqa_config_leaderboard.json'
RUN_MRC='/model/primeqa/mrc/run_mrc.py'

python ${RUN_MRC} \
  --model_name_or_path ${MRC_MODEL} \
  --output_dir ${OUTPUT_PATH} \
  --per_device_eval_batch_size 128 --overwrite_output_dir \
  --do_boolean --boolean_config ${BOOLEAN_CONFIG} \
  --max_eval_samples 100 --overwrite_cache



#python ${POSTPROCESSING_SCRIPT_LOCATION} --gold_path "${SYMLINKED_INPUT_PATH}" \
#                                         --predictions_path "${OUTPUT_DIR}/predictions.json" \
#                                         --output_path "${OUTPUT_PATH}"
