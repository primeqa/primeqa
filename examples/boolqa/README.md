# TydiQA with support for Boolean questions

The TydiQA dataset can be decoded, with full support for both the boolean and short-answer questions in two different ways.
As a single command line argument in `run_mrc.py` to run all steps:

```shell
python examples/mrc/run_mrc.py --model_name_or_path ${TRAINING_OUTPUT_DIR} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --do_boolean \
       --per_device_eval_batch_size 128 --overwrite_output_dir
```
or step-by-step.
The output of each individual step is analyzed in more detail [here](../../notebooks/boolqa/eval_predictions.ipynb)

## MRC

```shell
            python examples/mrc/run_mrc.py --model_name_or_path {mrcmodel} \
        --output_dir {ws}/mrc/ --fp16 --learning_rate 4e-5 \
        --do_eval --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
        --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
        --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no \
        --postprocessor oneqa.boolqa.processors.postprocessors.extractive.ExtractivePipelinePostProcessor
```

## QTC

```shell
    python examples/boolqa/run_nway_classifier_1qa.py \
    --task_name qtc \
    --overwrite_cache \
    --model_name_or_path {qtcmodel} \
    --test_file {mrcfile} \
    --output_dir {ws}/qtc
```
## EVC

```shell
    python examples/boolqa/run_nway_classifier_1qa.py \
    --task_name evc \
    --overwrite_cache \
    --drop_label no_answer \
    --model_name_or_path {evcmodel} \
    --test_file {ws}/qtc/eval_predictions.json \
    --output_dir {ws}/evc
```

## SN

```shell
    python examples/boolqa/merger_simple.py \
    --answer_predictions_file {ws}/evc/eval_predictions.json \
    --sn_model_file {sn_model_file} \
    --output_predictions_file {merge_prediction_file}
```
