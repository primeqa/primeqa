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