# TydiQA with support for Boolean questions

The TydiQA dataset can be decoded, with full support for both the boolean and short-answer questions in two different ways.
As a single command line argument in `run_mrc.py` to run all steps:

```shell
python examples/mrc/run_mrc.py --model_name_or_path ${TRAINING_OUTPUT_DIR} \
       --output_dir ${OUTPUT_DIR} --fp16 --do_eval \
       --do_boolean \
       --boolean_config ${BOOLEAN_CONFIG_FILE} \
       --per_device_eval_batch_size 128 --overwrite_output_dir
```
or step-by-step.
There are four stages in the process:
- **M**achine **R**eading **C**omprehension: given a question and and answer, find a representative span that may contain a short answer. This is analyzed in detail in the tydiqa.ipynb
- **Q**uestion **T**ype **C**lassification: given the question, decide if it is boolean or short_answer
- **Ev**idence **C**lassification: given a question and a short answer span, decide the short answer span supports yes or no. This is analyzed in more detail in evc.ipynb.
- **S**core **N**ormalization - span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to `[0,1]`.
The output of each individual step is analyzed in more detail this jupyter [notebook](../../notebooks/boolqa/eval_predictions.ipynb)

Some of this system has been described in ["Do Answers to Boolean Questions Need Explanations? Yes](https://arxiv.org/abs/2112.07772) 
and TODO

## Configuration file

The configuration file contains the parameters for each of the post-MRC steps
```
{
    "qtc": {
        "task_name": "qtc",
        "overwrite_cache": true,
        "model_name_or_path": ${QTC_MODEL_LOCATION},
        "test_file": "${MRC_OUTPUT_DIR}/eval_predictions.json",
        "output_dir": "${QTC_OUTPUT_DIR}"
    },
    "esc": {
        "task_name": "evc",
        "overwrite_cache": true,
        "max_seq_length": 500,
        "drop_label": "NONE",
        "model_name_or_path": ${EVC_MODEL_LOCATION},
        "test_file": "${QTC_OUTPUT_DIR}/eval_predictions.json",
        "output_dir": "${EVC_OUTPUT_DIR}
    },
    "sn": {
        "model_name_or_path": "${SN_MODEL_LOCATION}",
        "test_file": "${EVC_OUTPUT_DIR}/eval_predictions.json",
        "output_dir": "${SN_OUTPUT_DIR}"
    }
```
and consists of blocks that correspond to command line arguments of the individual steps (see below.)



## Machine Reading Comprehension

The machine reading comprehension differs from the default invocation of `run_mrc.py` (see [readme](../mrc/README.md))
in two ways.  First we provide a pretrained model that has been exposed to passage answer spans for boolean questions (the TydiQA
dataset does not provide short answers).  Second, the postprocessor provides additional information (language, question)
needed by the downstream components

```shell
python examples/mrc/run_mrc.py --model_name_or_path {mrcmodel} \
        --output_dir {ws}/mrc/ --fp16 --learning_rate 4e-5 \
        --do_eval --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
        --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
        --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no \
        --postprocessor oneqa.boolqa.processors.postprocessors.extractive.ExtractivePipelinePostProcessor
```

## Question type classification

Given a question (obtained from the `eval_predictions.json` file created in the previous step, predict
whether the question is `boolean` or `short_answer`.

```shell
    python examples/boolqa/run_nway_classifier_1qa.py \
    --task_name qtc \
    --overwrite_cache \
    --model_name_or_path {qtcmodel} \
    --test_file {mrcfile} \
    --output_dir {ws}/qtc
```
## Evidence classification

Given a question and the span predicted by the first step, predict whether the span supports
a `yes` or `no` answer to question.  Both question and span are passed through the `eval_predictions.json`
file output by the previous step.  The details of this process are analyzed in this jupyter [notebook](../../notebooks/boolqa/evc.ipynb).

```shell
    python examples/boolqa/run_nway_classifier_1qa.py \
    --task_name evc \
    --overwrite_cache \
    --drop_label no_answer \
    --model_name_or_path {evcmodel} \
    --test_file {ws}/qtc/eval_predictions.json \
    --output_dir {ws}/evc
```

## Score normalization

Span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to `[0,1]`.
and output a file suitable for the tydi evaluation script.

```shell
    python examples/boolqa/merger_simple.py \
    --answer_predictions_file {ws}/evc/eval_predictions.json \
    --sn_model_file {sn_model_file} \
    --output_predictions_file {merge_prediction_file}
```
