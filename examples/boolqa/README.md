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

### Train the Score Normalizer:

```
bash examples/boolqa/train_score_normalizer_for_tydi.sh
```

### Adapting the MRC component
We use as a starting point an MRC model trained on NQ, TyDi, and SQuad.  We then do 1 additional epoch of training
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

### Training the EVC component

First we subset the tydi data so that the EVC trainer only sees the answerable boolean questions:
```
python examples/boolqa/bool_tydi2csv.py --output_dir ${DATA_DIR}
```
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
