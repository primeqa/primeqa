## Preparing Data
We use "SQuAD" as the source dataset and "NewsQA" "NaturalQuestionsShort" "HotpotQA" "TriviaQA-web" as target datasets. All the datasets can be downloaded from [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019). We use the original dev set as the test set and sample a limited number (by default: 1000) of QA pairs from the training as the dev set.
Since there is no test set available for each dataset, we use the original dev set as the test set and sample 1,000 QA pairs from each target domain as the dev set (which we deem them as *target annotations* in the paper).
After preprocessing, you will have train/dev/test json files under ```data``` dir (by default) with the same format as [SQuAD](https://rajpurkar.github.io/SQuAD-explorer/).
```shell script
$ ./download_and_process.sh
```

## Base QG Model Training and Inference
We consider a semi-supervised domain adaptation setting where we first pretrain the QG model on the source domain 
and then finetune it on the limited number of target annotations (dev set).
And then we use the finetuned QG model to generate synthetic questions on all the target contexts. 
We finally convert the synthetic questions into the QA data format. 
```shell script
$ ./run_qg.sh
```

## Base QA Model Training
For QA model, we also first pretrain it on the source and then finetune it on the target synthetic and target dev.
```shell script
$ sh run_qa_baseline.sh
````

## QVE Training
We train Question Value Estimator (QVE) based on Reinforcement Learning (RL) to select the most useful QA pairs. 
To enable more stable RL training, we require large training batch size. 
But given the hardware GPU memory constraints, it is usually hard to have more than 16 examples in a batch (for 12GB GPU).
And due to the special reward calculation, it is hard to implement ```gradient accumulation```. 
To solve the issue, in our implementation, we offer two options:
1. Use smaller transformer (BERT) models (e.g., BERT-mini or BERT-small).
2. Use BERT-base but enable [```gradient checkpointing```](https://github.com/cybertronai/gradient-checkpointing), a technique used for reducing the memory footprint
when training deep neural networks.

Here we give two examples:

args description: 

- ``qa_model_name_or_path``: source trained QA model
- ``qve_model_name_or_path``: qve model
- ``marginal_model_name_or_path``: marginal QA model used to provide additional input to the QVE. 
It can be either source+target dev trained QA model or source trained QA model.
- ``train_file``: the target synthetic QA file
- ``dev_file``: the target dev file used to eval QA model and provide QVE reward
- ``do_train``: whether to train QVE model
- ``do_estimation``: whether to estimate the question value for all the questions in ``train_file``
- ``learning_rate``: qa model learning rate
- ``qve_learning_rate``: qve model learning rate
- ``reward_type``: what reward function to adopt: 'exact', 'f1' or 'loss'
- ``max_steps``: total training steps
- ``warmup_steps``: learning rate warmup steps (usually 10% of total training steps)
- ``add_marginal_info``: whether to add marginal info as additional input to QVE


```shell script
# Train with BERT-mini
python primeqa/examples/QVE/run_qve.py \
--qa_model_name_or_path checkpoints/QA_source_only \
--qve_model_name_or_path prajjwal1/bert-mini \
--marginal_model_name_or_path checkpoints/QA_TriviaQA-web_Source_TargetDev/ \
--do_lower_case \
--train_file data/TriviaQA-web_QG/TriviaQA-web.train.targetfinedtuned.gen.jsonl \
--dev_file data/TriviaQA-web.sample.dev.jsonl \
--do_train \
--do_estimation \
--per_gpu_train_qve_batch_size 64 \
--per_gpu_train_qa_batch_size 4 \
--learning_rate 3e-5 \
--qve_learning_rate 3e-5 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir checkpoints/TriviaQA-web_QVE_mini/ \
--overwrite_output_dir \
--logging_steps 5 \
--reward_type exact \
--max_steps 3000 \
--warmup_steps 300 \
--add_marginal_info
```

```shell script
# Train with BERT-base and gradient checkpointing
python primeqa/examples/QVE/run_qve.py \
--qa_model_name_or_path checkpoints/QA_source_only \
--qve_model_name_or_path bert-base-uncased \
--marginal_model_name_or_path checkpoints/QA_TriviaQA-web_Source_TargetDev/ \
--do_lower_case \
--train_file data/TriviaQA-web_QG/TriviaQA-web.train.targetfinedtuned.gen.json \
--dev_file data/TriviaQA-web.sample.dev.jsonl \
--do_train \
--per_gpu_train_qve_batch_size 80 \
--per_gpu_train_qa_batch_size 4 \
--learning_rate 3e-5 \
--qve_learning_rate 3e-5 \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir checkpoints/TriviaQA-web_QVE_base/ \
--overwrite_output_dir \
--logging_steps 5 \
--max_steps 1500 \
--warmup_steps 150 \
--gradient_checkpointing \
--add_marginal_info 
```

These two training strategies yield similar performance. Normally, our code can work well under *multi* GPU training.
But ``gradient checkpointing`` by the Pytorch does not work perfectly with *multi* GPU training (i.e., the training speed would become very slow!).
So we suggest turning off ``gradient checkpointing`` when you have enough GPU memory (e.g., train QVE with smaller transformers).


## QVE Inference (Select Useful Synthetic Subset)
After we train the QVE, we can use it to select the most useful (top K%) scored synthetic questions.
During the training, we automatically saved the best checkpoints based on: 
1) the highest reward; 2) the lowest QA training loss. You can use either the final trained model 
or the best checkpoint saved during training.

There is no dominating model selection strategy. Based on our observation, 
strategy (1) usually works better on NaturalQuestions;
strategy (2) usually works better on the HotpotQA and NewsQA datasets; 
the final trained model usually works better on the TriviaQA dataset. 

By default, we use the final trained model to do the selection:
```shell script
python primeqa/examples/QVE/run_qve.py \
--qa_model_name_or_path checkpoints/QA_source_only \
--qve_model_name_or_path checkpoints/TriviaQA-web_QVE_base \
--marginal_model_name_or_path checkpoints/QA_TriviaQA-web_Source_TargetDev/ \
--do_lower_case \
--train_file data/TriviaQA-web_QG/TriviaQA-web.train.targetfinedtuned.gen.jsonl \
--do_estimation \
--max_seq_length 384 \
--doc_stride 128 \
--output_dir checkpoints/TriviaQA-web_QVE_base/ \
--overwrite_output_dir \
--add_marginal_info \
--selected_question_percentage 0.6
```

The selected questions are saved in a json file (QA data format) under ``output_dir``. 
Finally, we train the QA model on the selected QA data:

```shell
python primeqa/mrc/run_mrc.py --model_name_or_path $SOURCE_QA_CKPT_DIR \
      --train_file checkpoints/TriviaQA-web_QVE_base/filtered_qa.jsonl \
      --eval_file data/TriviaQA-web.test.jsonl \
      --preprocessor primeqa.mrc.processors.preprocessors.squad.SQUADPreprocessor \
      --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
      --eval_metrics SQUAD \
      --max_seq_length 384 \
      --output_dir checkpoints/QA_TriviaQA-web_Source_Sythetic_QVEFiltering \
      --learning_rate 3e-5 \
      --do_train --do_eval --per_device_train_batch_size 12 \
      --per_device_eval_batch_size 64 \
      --save_steps 50000 \
      --overwrite_output_dir --num_train_epochs 2.0 \
      --overwrite_cache
```

