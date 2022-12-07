# This needs to be filled in.
output_dir = 'FILL_ME_IN'        # Save the results here.  Will overwrite if directory already exists.

# Optional parameters (feel free to leave as default).
model_name = 'roberta-base'  # Set this to select the LM.  Since this is a multi-lingual dataset, we use the XLM-Roberta model.
cache_dir = None                 # Set this if you have a cache directory for transformers.  Alternatively set the HF_HOME env var.
train_batch_size = 8             # Set this to change the number of features per batch during training.
eval_batch_size = 8              # Set this to change the number of features per batch during evaluation.
gradient_accumulation_steps = 8  # Set this to effectively increase training batch size.
max_train_samples = 100          # Set this to use a subset of the training data (or None for all).
max_eval_samples = 20            # Set this to use a subset of the evaluation data (or None for all).
num_train_epochs = 1             # Set this to change the number of training epochs.
fp16 = False                     # Set this to true to enable fp16 (hardware support required).
num_examples_to_show = 10 



from transformers import TrainingArguments
from transformers.trainer_utils import set_seed
from transformers import AutoConfig, AutoTokenizer
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks

from primeqa.mrc.trainers.mrc import MRCTrainer
from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from operator import attrgetter
import datasets
from transformers import DataCollatorWithPadding
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.metrics.squad import squad


import datasets
import random

seed = 42
set_seed(seed)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=train_batch_size,
    per_device_eval_batch_size=eval_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    evaluation_strategy='no',
    learning_rate=4e-05,
    warmup_ratio=0.1,
    weight_decay=0.1,
    save_steps=50000,
    fp16=fp16,
    seed=seed,
)

task_heads = EXTRACTIVE_HEAD
config = AutoConfig.from_pretrained(
    model_name,
    cache_dir=cache_dir,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=cache_dir,
    use_fast=True,
    config=config,
)
model = ModelForDownstreamTasks.from_config(
    config,
    model_name,
    task_heads=task_heads,
    cache_dir=cache_dir,
)
model.set_task_head(next(iter(task_heads)))

print(model)  # Examine the model structure

#dataset part 


raw_datasets = datasets.load_dataset(
    'squad',
    'plain_text',
    cache_dir=cache_dir,
)

train_examples = raw_datasets["train"]
max_train_samples = max_train_samples
if max_train_samples is not None:
    # We will select sample from whole data if argument is specified
    train_examples = train_examples.select(range(max_train_samples))

print(f"Using {train_examples.num_rows} train examples.")

eval_examples = raw_datasets["validation"]
max_eval_samples = max_eval_samples
if max_eval_samples is not None:
    # We will select sample from whole data if argument is specified
    random_idxs = random.sample(range(len(eval_examples)), max_eval_samples)
    eval_examples = eval_examples.select(random_idxs)

print(f"Using {eval_examples.num_rows} eval examples.")

#Preprocessing 

preprocessor = SQUADPreprocessor(
    stride=128,
    tokenizer=tokenizer,
)

# Train Feature Creation
with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_examples, train_dataset = preprocessor.process_train(train_examples)

print(f"Preprocessing produced {train_dataset.num_rows} train features from {train_examples.num_rows} examples.")

# Validation Feature Creation
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_examples, eval_dataset = preprocessor.process_eval(eval_examples)

print(f"Preprocessing produced {eval_dataset.num_rows} eval features from {eval_examples.num_rows} examples.")


# Finetuning the model on train dataset 

# If using mixed precision we pad for efficient hardware acceleration
using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

# noinspection PyProtectedMember
postprocessor = SQUADPostProcessor(
    k=3,
    n_best_size=20,
    max_answer_length=30,
    scorer_type=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF,
    single_context_multiple_passages=preprocessor._single_context_multiple_passages,
)

def compute_metrics(p: EvalPredictionWithProcessing):
    eval_metrics = datasets.load_metric(squad.__file__)
    return eval_metrics.compute(predictions=p.processed_predictions, references=p.label_ids)

trainer = MRCTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    eval_examples=eval_examples if training_args.do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    post_process_function=postprocessor.process_references_and_predictions,  # see QATrainer in Huggingface
    compute_metrics=compute_metrics,
)

train_result = trainer.train()
trainer.save_model()  # Saves the tokenizer too for easy upload

metrics = train_result.metrics
max_train_samples = max_train_samples or len(train_dataset)
metrics["train_samples"] = min(max_train_samples, len(train_dataset))

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#EVAL prediction analysis
metrics = trainer.evaluate()

max_eval_samples = max_eval_samples or len(eval_dataset)
metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)