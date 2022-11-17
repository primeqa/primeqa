# Few Shot Fine-tuning with ReasonBERT Models

This README provides instructions on how to fine-tune a extractive reader starting with [ReasonBERT](https://huggingface.co/osunlp/ReasonBERT-RoBERTa-base) pre-trained model in a few-shot setting.  

The datasets used for evaluation are subsets of the [MRQA](https://github.com/mrqa/MRQA-Shared-Task-2019) dataset specifically:

    - SQuAD
    - NaturalQuestionsShort
    - NewsQA 
    - TriviaQA-web
    - SearchQA

## What is ReasonBERT

ReasonBERT is a pretraining method that augments language models with the ability to reason over long-range relations and multiple contexts as described in [ReasonBERT: Pre-trained to Reason with Distant Supervision](https://arxiv.org/pdf/2109.04912).  As detailed in the paper, ReasonBERT proposes a distant supervision method to create pretrainig examples that require long range reasoning.  It pairs a query sentence with multiple relevant pieces of evidence drawn from possibly different places and defines a new LM pre-training objective, span reasoning, to recover entity spans that are masked out from the query sentence by jointly reasoning over the query sentence and the relevant evidence. 

Experiments on a variety of extractive question answering datasets show that ReasonBERT achieves strong performance compared to a variety of baselines.

Under the few-shot setting, ReasonBERT substantially outperforms a RoBERTa baseline on the extractive question answering task. 

Here we show how to replicate the results on the extractive question answering datasets using the [PrimeQA reader](../../primeqa/mrc/README.md). See section [Evalustion Scores](#evaluation-scores)

## Prerequisites
Before continuing below make sure you have PrimeQA [installed](../../README.md#Installation).

## Training and Evaluation

### Run Command

```
MODEL=osunlp/ReasonBERT-RoBERTa-base  # or roberta-base for baseline
OUTPUT_DIR=output
MRQA_SUBSET=SQuAD  # one of SQuAD NaturalQuestionsShort NewsQA TriviaQA-web SearchQA
NUM_EPOCHS=10
LR=5e-5
MSL=512    # 384 for SQuAD
STRIDE=384 # 128 for SQuAD
NUM_TRAIN_SAMPLES=128
SEED=234

python primeqa/mrc/run_mrc.py \
    --model_name_or_path ${MODEL} \
    --dataset_name mrqa --max_seq_length ${MSL} --doc_stride ${STRIDE} \
    --dataset_config_name plain_text \
    --dataset_filter_column_values ${MRQA_SUBSET} \
    --dataset_filter_column_name subset \
    --max_train_samples ${NUM_TRAIN_SAMPLES} --seed ${SEED} \
    --learning_rate ${LR} \
    --num_train_epochs ${NUM_EPOCHS} \
    --preprocessor primeqa.mrc.processors.preprocessors.mrqa.MRQAPreprocessor \
    --postprocessor primeqa.mrc.processors.postprocessors.squad.SQUADPostProcessor \
    --negative_sampling_prob_when_has_answer 1.0 \
    --negative_sampling_prob_when_no_answer 1.0 \
    --eval_metrics SQUAD \
    --output_dir ${OUTPUT_DIR} \
    --do_train --do_eval --fp16 \
    --per_device_train_batch_size 20 --per_device_eval_batch_size 128 \
    --gradient_accumulation_steps 1 --warmup_ratio 0.1 --weight_decay 0.01 \
    --save_steps 50000  \
    --evaluation_strategy no --single_context_multiple_passages
```

### SQuAD 

Run arguments:
```
MRQA_SUBSET=SQuAD  
NUM_TRAIN_SAMPLES=128
MODEL=osunlp/ReasonBERT-RoBERTa-base
NUM_EPOCHS=20
MSL=384 
STRIDE=128
SEED=234
NUM_TRAIN_SAMPLES=128
LR=5e-5
NUM_EPOCHS=50
```

Expected Results for ReasonBERT:
```
***** eval metrics *****
  epoch            =    20.0
  eval_exact_match = 55.2679
  eval_f1          = 64.5701
  eval_samples     =   10507
```

### NaturalQuestionsShort

Run arguments:
```
MODEL=osunlp/ReasonBERT-RoBERTa-base 
MRQA_SUBSET=NaturalQuestionsShort 
NUM_EPOCHS=10
LR=5e-5
MSL=512    
STRIDE=384 
NUM_TRAIN_SAMPLES=128
SEED=234
```

Expected Results for ReasonBERT
```
***** eval metrics *****
  epoch            =    10.0
  eval_exact_match = 38.2518
  eval_f1          = 48.2521
  eval_samples     =   12836
```


### NewsQA 

Run arguments:
```
MRQA_SUBSET=NewsQA 
NUM_EPOCHS=10
LR=5e-5
MSL=512    
STRIDE=384 
NUM_TRAIN_SAMPLES=128
SEED=234
```

Expected Results for ReasonBERT
```
***** eval metrics *****
  epoch            =    10.0
  eval_exact_match = 25.3324
  eval_f1          = 38.5748
  eval_samples     =    4212
```

### TriviaQA-web

Run arguments:
```
MRQA_SUBSET=TriviaQA-web 
NUM_EPOCHS=10
LR=5e-5
MSL=512    
STRIDE=384 
NUM_TRAIN_SAMPLES=128
SEED=123
```

Expected Output for ReasonBERT:
```
***** eval metrics *****
  epoch            =    10.0
  eval_exact_match = 49.7881
  eval_f1          = 55.0203
  eval_samples     =    7785
```


### SearchQA

Run arguments:
```
MRQA_SUBSET=SearchQA 
NUM_EPOCHS=10
LR=5e-5
MSL=512   
STRIDE=384 
NUM_TRAIN_SAMPLES=128
SEED=234
```

Expected Output for ReasonBERT:
```
***** eval metrics *****
  epoch            =    10.0
  eval_exact_match = 47.9682
  eval_f1          =  55.508
  eval_samples     =   16980
```

## Evaluation Scores

The following table shows F1 scores on subsets of the MRQA dataset in full and few-shot settings for the baseline `roberta-base` and for `osunlp/ReasonBERT-RoBERTa-base`.  The scores are average of four runs with different seeds.

Under the full-data setting, ReasonBERT achieves accuracy similar to the baseline pretrained model RoBERTa.
Under the few-shot setting (128 examples), ReasonBERT outperforms a RoBERTa base model by a large margin in all cases.

Model | Train Size | SQuAD | TriviaQA | NQ | NewsQA | SearchQA
-- | -- | -- | -- | -- | -- | --
RoBERTa  | all | 90.0 | 74.4 | 79.1 | 71.7 | 79.6
ReasonBERT  | all | 88.8 | 74.5 | 78.5 | 69.3 | 79.0
RoBERTa | 128 | 50.9+-2.1 | 21+-2.2 | 33.9+-1.5 | 23.2+-3.5 | 28.6+-1.5
ReasonBERT | 128 | **64.3+-0.8** | **54.1+-0.9** | **47.3+-0.3** | **36.6+-2.0** | **53.4+-2.4**
RoBERTa  | 1024 | 76.6+-0.4 | 49+-1.5 | 59.8+-0.9 | 53.7+-0.9 | 55.7+-1.7
ReasonBERT | 1024 | 77.1+-0.4 | 60+-0.9 | 64+-0.2 | 54.3+-0.5 | 64.2+-0.7

NOTE: the score may not exactly match the results in the paper due to some differences in the reader implementation.


## Citation

```
@inproceedings{deng-etal-2021-reasonbert,
    title = "{R}eason{BERT}: {P}re-trained to Reason with Distant Supervision",
    author = "Deng, Xiang  and
      Su, Yu  and
      Lees, Alyssa  and
      Wu, You  and
      Yu, Cong  and
      Sun, Huan",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.494",
    doi = "10.18653/v1/2021.emnlp-main.494",
    pages = "6112--6127",
}
```