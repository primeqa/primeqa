# TydiQA with support for Boolean questions

The TydiQA dataset can be decoded, with full support for both the boolean and short-answer questions in two different ways.
As a single command line argument in `run_mrc.py` to run all steps:

```shell
python examples/mrc/run_mrc.py --model_name_or_path ${BOOLEAN_MODEL_NAME} \
       --output_dir ${OUTPUT_DIR} --fp16 \
       --per_device_eval_batch_size 128 --overwrite_output_dir \
       --do_boolean --boolean_config  examples/boolqa/tydi_boolqa_config.json
```
or step-by-step.
The option `--do_boolean` supercedes the `--do_eval` option, and runs the following four-stage process:

- **M**achine **R**eading **C**omprehension: given a question and and answer, find a representative span that may contain a short answer. This is analyzed in detail in the tydiqa.ipynb
- **Q**uestion **T**ype **C**lassification: given the question, decide if it is boolean or short_answer
- **Ev**idence **C**lassification: given a question and a short answer span, decide the short answer span supports yes or no. This is analyzed in more detail in evc.ipynb.
- **S**core **N**ormalization - span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to `[0,1]`.

### Jupyter notebooks
The output of each individual step is analyzed in more detail this jupyter [notebook](../../notebooks/boolqa/eval_predictions.ipynb).

The inner details of the evidence classifier are analyzed in more detail in [notebook](../../notebooks/boolqa/evc.ipynb).

Some of this system has been described in ["Do Answers to Boolean Questions Need Explanations? Yes](https://arxiv.org/abs/2112.07772) 
and TODO

## Configuration file

The configuration file contains the parameters for each of the post-MRC steps
```
{
    "qtc": {
        "id_key": "example_id",
        "sentence1_key": "question",
        "sentence2_key": null,
        "label_list": ["short_answer","boolean"],
        "output_label_prefix": "question_type",
        "overwrite_cache": true,
        "use_auth_token": true,
        "model_name_or_path": "ibm/qtc_bert_pretrained_model"
    },
    "evc": {
        "id_key": "example_id",
        "sentence1_key": "question",
        "sentence2_key": "passage_answer_text",
        "label_list": ["no", "no_answer", "yes"],
        "output_label_prefix": "boolean_answer",
        "overwrite_cache": true,
        "drop_label": "no_answer",
        "use_auth_token": true,
        "model_name_or_path": "ibm/evc_xlm_roberta_large"
    },
    "sn": {
        "model_name_or_path": "tests/resources/boolqa/score_normalizer_model/sn.pickle",
        "qtc_is_boolean_label": "boolean",
        "evc_no_answer_class": "no_answer"
    }
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
        --postprocessor primeqa.boolqa.processors.postprocessors.extractive.ExtractivePipelinePostProcessor
```

## Question type classification

Given a question (obtained from the `eval_predictions.json` file created in the previous step, predict
whether the question is `boolean` or `short_answer`.

```shell
python examples/boolqa/run_boolqa_classifier.py \
    --overwrite_cache \
    --id_key example_id \
    --sentence1_key question \
    --label_list short_answer boolean \
    --output_label_prefix question_type \
    --model_name_or_path ibm/qtc_bert_pretrained_model \
    --test_file ${BASE}/eval_predictions.json \
    --output_dir ${OUTDIR}/qtc \
    --use_auth_token
```
## Evidence classification

Given a question and the span predicted by the first step, predict whether the span supports
a `yes` or `no` answer to question.  Both question and span are passed through the `eval_predictions.json`
file output by the previous step.  The details of this process are analyzed in this jupyter [notebook](../../notebooks/boolqa/evc.ipynb).

```shell
python examples/boolqa/run_boolqa_classifier.py \
    --overwrite_cache  \
    --id_key example_id \
    --sentence1_key question \
    --sentence2_key passage_answer_text \
    --label_list no no_answer yes \
    --output_label_prefix boolean_answer \
    --drop_label no_answer \
    --model_name_or_path ibm/evc_xlm_roberta_large \
    --test_file ${BASE}/qtc/eval_predictions.json \
    --output_dir ${OUTDIR}/evc \
    --use_auth_token
```

## Score normalization

Span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to `[0,1]`.
and output a file suitable for the tydi evaluation script.

```shell
python examples/boolqa/run_score_normalizer.py \
    --test_file ${BASE}/evc/eval_predictions.json \
    --model_name_or_path tests/resources/boolqa/score_normalizer_model/sn.pickle \
    --output_dir ${OUTDIR}/sn
```
