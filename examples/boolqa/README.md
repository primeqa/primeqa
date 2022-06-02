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
There are four stages in the process:
MRC (machine reading comprehension) - given a question and and answer, find a representative span that may contain a short answer. This is analyzed in detail in the tydiqa.ipynb
QTC (question type classifier) - given the question, decide if it is boolean or short_answer
EVC (evidence classifier) - given a question and a short answer span, decide the short answer span supports yes or no. This is analyzed in more detail in evc.ipynb.
Score normalization - span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to $[0,1]$.
The output of each individual step is analyzed in more detail this jupyter [notebook](../../notebooks/boolqa/eval_predictions.ipynb)

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

Span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to $[0,1]$.
and output a file suitable for the tydi evaluation script.

```shell
    python examples/boolqa/merger_simple.py \
    --answer_predictions_file {ws}/evc/eval_predictions.json \
    --sn_model_file {sn_model_file} \
    --output_predictions_file {merge_prediction_file}
```
