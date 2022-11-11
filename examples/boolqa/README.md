# Training a full TyDiQA model

Here we describe how to train the TyDiQA model `PrimeQA-Reader-with-Boolean` which was submitted to the TyDi leaderboard on 11/1/2022.
This model has full support for boolean questions.  For simplified inference-only TyDiQA with suppport for boolean questions using earlier models,
please see [here](../../primeqa/boolqa/README.md).
Training this model is a multistage process:

- first we customize a general-purpose MRC model to Tydi

- then we train a question type classifier that determines if questions are short answer or yes/no.

- then we train an evidence classifier that decides how the yes/no quesitons should be answered

- then we train a score normalizer so that the answerable/non-answerable distinctions are reflected in the confidence score,
independent of question type.

- Finally we integrate these models into the configuration file so that the `do_boolean` option in `run_mrc.py` can use them.

## Description

Instructions for generating csv files for boolean question type and evidence span classification from the [TyDi HF]() dataset. The outputted files can be used in companion with [run_nway_classifier.py](../../primeqa/text_classification/run_nway_classifier.py) (see [README](../../primeqa/text_classification/README.md)) for classification. 

```
TRAIN_OUTPUT_FILE = /location/to/save/output.csv

PrimeQA/examples/boolqa/bool_tydi2csv.py
        --output_dir $TRAIN_OUTPUT_DIR
```

There is also the option to save all text in lower case using the `--lower_case` argument.

This will dump train and dev files for question type (`qtype_*.csv`) and evidence span (`evidence_span_*.csv`). The outputted files are formatted as follows:

Question Type:

|example_id|question|language|label|
| :---: | :---: | :---: | :---: |
|166917|หม่อมราชวงศ์สุขุมพันธุ์ บริพัตร เรียนจบจากที่ไหน ?|thai|other|
|166918|Ukubwa wa Rijili Kantori ni kiasi gani?|swahili|other|

Evidence Span:

|example_id|question|language|label|passage|
| :---: | :---: | :---: | :---: | :---: |
|166917|หม่อมราชวงศ์สุขุมพันธุ์ บริพัตร เรียนจบจากที่ไหน ?|thai|NONE| หมวดหมู่:หม่อมราชวงศ์ หมวดหมู่:ราชสกุลบริพัตร หมวดหมู่:นักการเมืองไทย สุขุมพันธุ์ หมวดหมู่:พรรคประชาธิปัตย์ หมวดหมู่:รองศาสตราจารย์ หมวดหมู่:อาจารย์คณะรัฐศาสตร์ หมวดหมู่:บุคคลจากคณะรัฐศาสตร์ จุฬาลงกรณ์มหาวิทยาลัย หมวดหมู่:สมาชิกเครื่องราชอิสริยาภรณ์ ม.ป.ช. หมวดหมู่:สมาชิกเครื่องราชอิสริยาภรณ์ ม.ว.ม. หมวดหมู่:สมาชิกเครื่องราชอิสริยาภรณ์ ต.จ. (ฝ่ายหน้า) หมวดหมู่:สมาชิกเครื่องราชอิสริยาภรณ์ บ.ภ. หมวดหมู่:บุคคลจากมหาวิทยาลัยออกซฟอร์ด หมวดหมู่:สมาชิกสภาผู้แทนราษฎรกรุงเทพมหานคร หมวดหมู่:สมาชิกสภาผู้แทนราษฎรไทยแบบสัดส่วน หมวดหมู่:สมาชิกสภาผู้แทนราษฎรไทยแบบบัญชีรายชื่อ หมวดหมู่:ชาวไทยเชื้อสายมลายู  หมวดหมู่:ราชสกุลปาลกะวงศ์|
|166918|Ukubwa wa Rijili Kantori ni kiasi gani?|swahili|NONE|"Proxima Centauri (yaani nyota ya Kantarusi iliyo karibu zaidi nasi) imegunduliwa kuwa na sayari moja. Vipimo vinavyopatikana hadi sasa zinaonyesha uwezekano mkubwa ya kwamba sayari hii ni ya mwamba (kama dunia yetu, Mirihi au Zuhura) na inaweza kuwa na angahewa, tena katika upeo wa joto unaoruhusu kuwepo kwa uhai. [1]"|

### Adapting the MRC component
We use as a starting point an MRC model trained on NQ, TyDi, and SQuad.  Support for training this model is coming soon.
We then do 1 additional epoch of training
with TyDi.  Since Tydi does not provide minimal answer begin and ends for boolean questions, we use a custom preprocessor 
that maps the passage answer begin and ends as the reference for training examples.  We will not need this preprocessor at inference time.
```
epochs=1
seed=42
lr=1e-5
python primeqa/mrc/run_mrc.py \
  --model_name_or_path  ${BASE_MODEL} \
  --output_dir ${OUTPUT_DIR} --fp16 --learning_rate ${lr} \
  --do_train \
  --seed ${seed} \
  --do_eval --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 128 --gradient_accumulation_steps 4 \
  --warmup_ratio 0.1 --weight_decay 0.1 --save_steps -1 \
  --overwrite_output_dir \
  --num_train_epochs ${epochs} \
  --preprocessor primeqa.mrc.processors.preprocessors.tydiboolqa_bpes.TyDiBoolQAPreprocessor \
  --evaluation_strategy no
```
This procedure yielded `eval_avg_minimal_f1=0.7006`.

### Training the QTC and EVC components

First we subset the tydi data so that the QTC trainer only sees the answerable questions and the EVC trainer only sees the answerable boolean questions:
```
python examples/boolqa/bool_tydi2csv.py --output_dir ${DATA_DIR}
```
This script will output files for QTC (`qtype_*.csv`) and EVC (`evidence_span_*.csv`) for train, validation, and testing. The test files will have all questions (including unanswerable) in the dev set without labels.

#### QTC

```
seed=42
epochs=3
lr=5e-5
python /primeqa/text_classification/run_nway_classifier.py \
       --model_name_or_path xlm-roberta-large \
       --do_train \
       --do_eval \
       --train_file ${DATA_DIR}/qtype_train.csv \
       --validation_file ${DATA_DIR}/qtype_val.csv \
       --output_dir ${OUTPUT_DIR} \
       --balanced \
       --output_label_prefix 'qtype' \
       --per_device_eval_batch_size 128 \
       --per_device_train_batch_size 64 \
       --save_steps -1 \
       --sentence1_key "question" \
       --label_list "boolean" "other" \
       --num_train_epochs ${epochs} \
       --learning_rate ${lr} \
       --seed ${seed}

```
This yielded f-measures of 0.969 on the boolean questions, and 0.996 on the short answer questions.


#### EVC 

Then we train the classifier
```
epochs=10
seed=1235
lr=1e-5
python primeqa/text_classification/run_nway_classifier.py \
  --overwrite_cache \
  --train_file ${DATA_DIR}/evidence_span_train.csv \
  --validation_file ${DATA_DIR}/evidence_span_val.csv \
  --model_name_or_path xlm-roberta-large \
  --do_train \
  --do_eval \
  --learning_rate ${lr} \
  --num_train_epochs ${epochs} \
  --max_seq_length 500 \
  --output_dir ${OUTPUT_DIR} \
  --save_steps -1 \
  --seed ${seed} \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --overwrite_output_dir \
  --logging_dir ${OUTPUT_DIR}/log/ \
  --sentence2_key passage \
  --label_list NO YES \
  --fp16 \
  --output_label_prefix evc
```
This yielded f-measures of 0.6 on the NO questions, and 0.93 on the YES questions.


### Training the Score Normalizer:

The following script will split the training data into to two parts to train and evaluate the score normalizer (90%/10%). It will run `run_mrc.py` with `--do_boolean` on the training portion (90%) to get the scores produced from `run_mrc.py` and the question type labels (`boolean`, `other`). Finally, the scores and question labels will be used to train the the score normalizer and evaluate on the remaining portion of the training set (10%). The final output will be stored in the `run_mrc.py` prediction file in `output_dir/sn`.

```
bash examples/boolqa/train_score_normalizer_for_tydi.sh
```
