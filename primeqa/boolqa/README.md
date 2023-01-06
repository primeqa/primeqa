<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.boolqa

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# TyDiQA with support for Boolean questions

Here, we describe how PrimeQA supports both the boolean questions and the short answer questions in the TyDiQA dataset. This
description is inference-only, assuming existing models. For a description of how such models are trained as part of the 
11/1/2022 TyDiQA leaderboard submission, see the [README](../../extensions/boolqa/README.md) 

The TyDiQA dataset can be decoded, with full support for both boolean and short-answer questions in two different ways:
First, as a single command line argument in `run_mrc.py` to run all steps, and secondly, as a step-by-step process detailed in the 
later sections of this README.  
The single command line argument is:
```shell
python primeqa/mrc/run_mrc.py --model_name_or_path PrimeQA/tydi-reader_bpes-xlmr_large-20221117 \
       --output_dir ${OUTPUT_DIR} --fp16 --overwrite_cache \
       --per_device_eval_batch_size 128 --overwrite_output_dir \
       --do_boolean --boolean_config  primeqa/boolqa/tydi_boolqa_config.json
```
The option `--do_boolean` supercedes the `--do_eval` option, and runs the following four-stage process:

- **M**achine **R**eading **C**omprehension: given a question and answer, find a representative span that may contain a short answer. This is analyzed in detail [here](https://github.com/primeqa/primeqa/blob/main/notebooks/mrc/tydiqa.ipynb)
- **Q**uestion **T**ype **C**lassification: given the question, decide if it is boolean or short_answer
- Answer classification (or **Ev**idence **C**lassification): given a question and an answer span, decide whether the span supports yes or no. 
- **S**core **N**ormalization - span scores may have different dynamic ranges according to whether the question is boolean or short_anwer. Normalize them uniformally to `[0,1]`.

The final evaluation of results after these four steps is in `${OUTPUT_DIR}/sn/all_results.json`.  The evaluation after only the MRC step is in `${OUTPUT_DIR}/eval_results.json`.

We provide pretrained models for each of these downstream components.

### Jupyter notebooks and further details

The output of each individual step is analyzed in more detail this jupyter [notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/boolqa/eval_predictions.ipynb).

The inner details of the machine reading comprehension for TydiQA are analyzed in more detail in [notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/mrc/tydiqa.ipynb).

The inner details of the answer classifier are analyzed in more detail in [notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/boolqa/evc.ipynb).

Some of this system has been described in the papers [Do Answers to Boolean Questions Need Explanations? Yes](https://arxiv.org/abs/2112.07772) 
and [GAAMA 2.0: An Integrated System that Answers Boolean and Extractive Questions](https://arxiv.org/abs/2206.08441)

## Configuration file

The configuration file contains the parameters for each of the post-MRC steps
```
{
    "qtc": {
        "id_key": "example_id",
        "sentence1_key": "question",
        "sentence2_key": null,
        "label_list": ["boolean", "other"],
        "output_label_prefix": "question_type",
        "overwrite_cache": true,
        "use_auth_token": true,
        "model_name_or_path": "PrimeQA/tydi-tydi_boolean_question_classifier-xlmr_large-20221117"
    },
    "evc": {
        "id_key": "example_id",
        "sentence1_key": "question",
        "sentence2_key": "passage_answer_text",
        "label_list": ["no", "yes"],
        "output_label_prefix": "boolean_answer",
        "overwrite_cache": true,
        "use_auth_token": true,
        "model_name_or_path": "PrimeQA/tydi-tydi_boolean_answer_classifier-xlmr_large-20221117"
    },
    "sn": {
	"do_apply": true,
        "model_name_or_path": "tests/resources/boolqa/score_normalizer_model/sn.pickle",
        "qtc_is_boolean_label": "boolean",
        "evc_no_answer_class": "no_answer"
    }
}
```
and consists of blocks for each of the downstream components.  The individual arguments correspond to command line arguments of the individual steps (see below.)



## Machine Reading Comprehension

The machine reading comprehension differs from the default invocation of `run_mrc.py` (see [readme](../../api/mrc/index))
as follows: the postprocessor provides additional information (language, question)
needed by the downstream components

```shell
python primeqa/mrc/run_mrc.py --model_name_or_path PrimeQA/tydi-reader_bpes-xlmr_large-20221117 \
        --output_dir ${BASE}/mrc/ --fp16 --learning_rate 4e-5 \
        --do_eval --per_device_train_batch_size 16 --fp16 \
        --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
        --warmup_ratio 0.1 --weight_decay 0.1 --save_steps 50000 \
        --overwrite_output_dir --num_train_epochs 1 --evaluation_strategy no \
        --postprocessor primeqa.text_classification.processors.postprocessors.extractive.ExtractivePipelinePostProcessor
```

## Question type classification

Given a question (obtained from the `eval_predictions.json` file created in the previous step, predict
whether the question is `boolean` or `short_answer`.

```shell
python primeqa/text_classification/run_nway_classifier.py \
    --overwrite_cache \
    --example_id_key example_id \
    --do_mrc_pipeline \
    --fp16 \
    --sentence1_key question \
    --label_list boolean other \
    --output_label_prefix question_type \
    --model_name_or_path PrimeQA/tydi-tydi_boolean_question_classifier-xlmr_large-20221117 \
    --test_file ${BASE}/mrc/eval_predictions.json \
    --output_dir ${BASE}/qtc \
    --use_auth_token
```
## Answer classification

Given a question and the passage predicted by the first step, predict whether the span supports
a `yes` or `no` answer to question.  Both question and span are passed through the `eval_predictions.json`
file output by the previous step.  The details of this process are analyzed in this jupyter [notebook](https://github.com/primeqa/primeqa/blob/main/notebooks/boolqa/evc.ipynb).

```shell
python primeqa/text_classification/run_nway_classifier.py \
    --overwrite_cache  \
    --example_id_key example_id \
    --do_mrc_pipeline \
    --sentence1_key question \
    --sentence2_key passage_answer_text \
    --label_list no yes \
    --output_label_prefix boolean_answer \
    --model_name_or_path PrimeQA/tydi-tydi_boolean_answer_classifier-xlmr_large-20221117 \
    --test_file ${BASE}/qtc/predictions.json \
    --output_dir ${BASE}/evc \
    --use_auth_token
```

## Score normalization

Span scores may have different dynamic ranges according as whether the question is boolean or short_anwer. Normalize them uniformally to `[0,1]`.
and output a file suitable for the TyDiQA evaluation script.
Warning: The score normalizer was developed for a leaderboard submission with a hidden test set.

Therefore, the score normalizer component was trained on system output on half of the dev set.  As a result the full dev set results are contaminated.  The fair evaluation of the submission is on the TyDiQA [leaderboard](https://ai.google.com/research/tydiqa) itself.

Another fair evalutation is on the other half of the dev set - for details, see [training](../../extensions/boolqa/README.md) 


```shell
python primeqa/boolqa/run_score_normalizer.py \
    --test_file ${BASE}/evc/predictions.json \
    --model_name_or_path tests/resources/boolqa/score_normalizer_model/sn.pickle \
    --output_dir ${BASE}/sn \
    --do_apply
```
