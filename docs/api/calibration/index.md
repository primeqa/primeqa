<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.calibration

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# Machine Reading Comprehension with Confidence Calibration

## train_confidence_calibrator.py     

This script trains a confidence calibration model which can be 
used by the run_mrc pipeline to generate a confidence score for each answer.

In the script, a confidence calibration model is built in multiple
steps:
- Split the train set of raw dataset, e.g. tydiqa, into mrc_train_set
  and confidence_train_set with the ratio specified by "relative_confidence_train_size".
  The new dataset, including the two train sets and original
  validation set is saved to the directory "confidence_dataset_dir".
- Train MRC model with mrc_train_set.
- Run MRC model in eval mode on confidence_train_set to generate confidence
  features by using the task head "EXTRACTIVE_WITH_CONFIDENCE_HEAD".
- Train confidence calibration model using the features from
  confidence_train_set, and save the result to the directory
  "confidence_model_dir".

The script also runs the trained MRC model on validation set to predict 
an answer for each question, and calculates the confidence score by using the 
confidence calibration model. The result is saved to the file 
"eval_predictions.rescored.json" in the directory "output_dir", 
which can be used to evaluate the performance of the confidence calibration model.

### Supported Datasets
Currently supported datasets include:
- TyDiQA

### Example Usage
An example usage using TyDiQA dataset is as follows.
```shell
python primeqa/calibration/train_confidence_calibrator.py \
       --model_name_or_path xlm-roberta-large \
       --dataset_name tydiqa \
       --output_dir ${OUTPUT_DIR} \
       --fp16 --learning_rate 4e-5 \
       --do_train --do_eval \
       --per_device_train_batch_size 16 \
       --per_device_eval_batch_size 32 \
       --gradient_accumulation_steps 4 \
       --warmup_ratio 0.1 --weight_decay 0.1 \
       --save_steps 50000 \
       --overwrite_output_dir --num_train_epochs 1 \
       --evaluation_strategy no \
       --confidence_dataset_dir ${CONFIDENCE_DATASET_DIR} \
       --relative_confidence_train_size 0.1 \
       --output_dropout_rate 0.25 \
       --decoding_times_with_dropout 5 \
       --confidence_model_dir ${CONFIDENCE_MODEL_DIR} \
       --max_iter_of_confidence_model_training 200 \
       --prediction_reference_overlap_threshold 0.5 \
       --task_heads primeqa.mrc.models.heads.extractive.EXTRACTIVE_WITH_CONFIDENCE_HEAD
```
### Output
- Confidence calibration model file "confidence.bin" saved in the 
directory "confidence_model_dir".
- Reorganized dataset that the Confidence calibration model is built on.
The dataset is saved in "confidence_dataset_dir".
- Prediction file of validation set including confidence score. The file
"eval_predictions.rescored.json" is saved in the directory "output_dir".
