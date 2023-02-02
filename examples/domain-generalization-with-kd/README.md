This directory contains an implementation of multi-source domain generalization via knowledge distillation for SQuAD-style reading comprehension tasks, as proposed in the following EMNLP 2022 paper:

```
Not to Overfit or Underfit the Source Domains? An Empirical Study of Domain Generalization in Question Answering
Md Arafat Sultan, Avirup Sil and Radu Florian
```
https://aclanthology.org/2022.emnlp-main.247/

Knowledge distillation is a model supervision technique where the model we want to train -- the 'student' -- learns a task (text-based reading comprehension in this case) from a different, typically larger and higher-accuracy model, called the 'teacher'. See https://arxiv.org/abs/1503.02531 for more details.

To train and validate a model with simple joint training over multiple source domains (no distillation), run the following command:
```
python primeqa/mrc/run_mrc.py \
       --model_name_or_path bert-large-uncased \
       --output_dir <path-to-output-dir> \
       --fp16 \
       --do_train \
       --task_trainer primeqa.mrc.trainers.mrc_mskd.MSKD_MRCTrainer \
       --train_fof <path-to-file-containing-train-file-locations> \
       --learning_rate 3e-5 \
       --per_device_train_batch_size 16 \
       --num_train_epochs 2 \
       --negative_sampling_prob_when_has_answer 0.0 \
       --warmup_ratio 0.1 \
       --weight_decay 0.1 \
       --save_strategy epoch \
       --do_eval \
       --eval_fof <path-to-file-containing-validation-file-locations> \
       --per_device_eval_batch_size 128 \
       --eval_metrics SQUAD \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --overwrite_output_dir \
       --overwrite_cache
```
This basic joint training functionality is offered through this source file so that one can train a teacher model first before distillation.

The argument for `train_fof` in the command is a text file; each line in this file contains a path to a training json file. All training files must conform to the Hugging Face Datasets format for SQuAD-style data files. The `eval_fof` argument works similarly. Below is an example of the content of such a file:
```
<path-to-train-data-dir>/SQuAD-hf.json
<path-to-train-data-dir>/NaturalQuestions-hf.json
<path-to-train-data-dir>/NewsQA-hf.json
```
It points to three training files to be used in the experiment. The `-hf` suffix is there as a reminder that the files must be in the Hugging Face Datasets SQuAD format.

The BERT-large QA model that the above command trains can then be used as a teacher in a distillation experiment to train a BERT-base student, as follows:
```
python primeqa/mrc/run_mrc.py \
       --model_name_or_path bert-base-uncased \
       --output_dir <path-to-student-output-dir> \
       --fp16 \
       --do_train \
       --task_trainer primeqa.mrc.trainers.mrc_mskd.MSKD_MRCTrainer \
       --train_fof <path-to-file-containing-train-file-locations> \
       --learning_rate 3e-5 \
       --per_device_train_batch_size 16 \
       --num_train_epochs 2 \
       --kd_teacher_model_path <path-to-trained-kd-teacher>/pytorch_model.bin \
       --kd_teacher_config_path <path-to-trained-kd-teacher>/config.json \
       --kd_temperature 2. \
       --negative_sampling_prob_when_has_answer 0.0 \
       --warmup_ratio 0.1 \
       --weight_decay 0.1 \
       --save_strategy epoch \
       --do_eval \
       --eval_fof <path-to-file-containing-validation-file-locations> \
       --per_device_eval_batch_size 128 \
       --eval_metrics SQUAD \
       --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
       --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
       --overwrite_output_dir \
       --overwrite_cache       
```

We also provide a script for the conversion of MRQA 2019-style datasets (https://github.com/mrqa/MRQA-Shared-Task-2019) to the Hugging Face Datasets SQuAD format:
```
python utils/convert_to_hf_format.py <path-to-input-mrqa-style-file> <path-to-output-hf-style-file>
```
The first argument is the path to the input input file (in MRQA format) and the second argument is the output file path which will contain the data in HF Datasets SQuAD-style format.