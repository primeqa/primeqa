# in this file, load NQ and preprocess to save offline
import datasets
from primeqa.mrc.processors.preprocessors.natural_questions import NaturalQuestionsPreProcessor
from datasets import disable_caching, concatenate_datasets, load_dataset
from primeqa.mrc.run_mrc_utils import object_reference, get_raw_datasets, process_raw_datasets
from transformers import HfArgumentParser, TrainingArguments, AutoConfig, AutoTokenizer
from glob import glob
import sys
import math
import itertools
import os

dataset_name = "natural_questions"
dataset_config_name = "default"
beam_runner = "DirectRunner"
preprocessing_num_workers = 5
input_dir = "/dccstor/srosent3/primeqa/data/nq_adapted/"
output_dir = "/dccstor/srosent3/primeqa/data/nq-tokenized/"
split = '0' #sys.argv[1]
pad_on_right = True
stride = 256
max_seq_length = None
max_q_char_len = 128
tokenizer = None

do_train = False
adapt = True
tokenize = False
load = False

# dataset = load_dataset("arrow", data_files={'train': '/dccstor/srosent3/primeqa/data/nq_adapted/train/subset20492-40984/dataset.arrow'})

if tokenize:
    config = AutoConfig.from_pretrained(
        'xlm-roberta-large'
    )
    tokenizer = AutoTokenizer.from_pretrained(
    'xlm-roberta-large',
    use_fast=True,
    config=config,
    )


preprocessor = NaturalQuestionsPreProcessor(tokenizer=tokenizer, num_workers=preprocessing_num_workers, load_from_cache_file=False) 

if adapt:
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        do_train=False,
        do_eval=True,
    )
    # turn off caching
    raw_datasets = datasets.load_dataset(
                        dataset_name,
                        dataset_config_name,
                        beam_runner=beam_runner,
                        revision="main"
                    )
    disable_caching()

    train_size = len(raw_datasets['train'])
    eval_size = len(raw_datasets['validation'])
    print(f'train size: {train_size}')
    print(f'eval size: {eval_size}')
    
    if do_train:
        train_split_size = math.ceil(train_size / 15)
        start = math.floor(int(split) * train_split_size)
        end = start + train_split_size
        if end > train_size:
            end = train_size
        print(f"{start}:{end}")
    else:
        start = 0
        end = eval_size

    if do_train:
        with training_args.main_process_first(desc=f"train dataset map pre-processing"):
            train_examples = raw_datasets['train'].select(range(start, end))
            processed_train_dataset = preprocessor.adapt_dataset(train_examples,True)
        processed_train_dataset.save_to_disk(output_dir + f"/train/subset{start}-{end}")
    else:
        # process val data
        with training_args.main_process_first(desc=f"eval dataset map pre-processing"):
            eval_examples = raw_datasets['validation'].select(range(start, end))
            processed_eval_dataset = preprocessor.adapt_dataset(eval_examples,False)
        processed_eval_dataset.save_to_disk(output_dir + f"/eval/subset{start}-{end}")

if load:
    train_files = glob(output_dir + "/train/*")
    eval_files = glob(output_dir + "/eval/*")

    train_datasets = [] 
    for file in train_files:
        d = datasets.load_from_disk(file)
        print(len(d))
        train_datasets.append(d)
    train_dataset = concatenate_datasets(train_datasets)
    print(len(train_dataset))
    
    eval_datasets = [] 
    for file in eval_files:
        d = datasets.load_from_disk(file)
        print(len(d))
        eval_datasets.append(d)
    eval_dataset = concatenate_datasets(eval_datasets)
    print(len(eval_dataset))

if tokenize:        
    split_type = "train"

    examples = datasets.load_from_disk(input_dir + f"/{split_type}/subset{split}")
    examples = examples #.select(range(200))

    examples_question = examples['question']
    examples_context = examples['context']
    if isinstance(examples_question, str):  # wrap single (question, [context]) pair in list
        examples_question = [examples_question]
        examples_context = [examples_context]
    examples_question = [q.lstrip()[:max_q_char_len] for q in examples_question]
    
    # create 1:1 question:context lists
    expanded_examples_question = []
    expanded_examples_idx = []
    for i, (question, context) in enumerate(zip(examples_question, examples_context)):
        context = preprocessor._trim_to_max_contexts(context, examples, i)
        n_context_for_example = len(context)
        
        expanded_examples_question.extend(itertools.repeat(question, n_context_for_example))
        expanded_examples_idx.extend(itertools.repeat(i, n_context_for_example))
    expanded_examples_context = list(itertools.chain.from_iterable(examples_context))

    tokenized_examples = preprocessor._tokenizer(
        expanded_examples_question if pad_on_right else expanded_examples_context,
        expanded_examples_context if pad_on_right else expanded_examples_question,
        stride=stride,
        max_length=max_seq_length,
        truncation='only_second' if pad_on_right else 'only_first',
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
    )

    tokenized_examples['example_idx'] = [expanded_examples_idx[oidx] for oidx in
                                        tokenized_examples["overflow_to_sample_mapping"]]
    tokenized_examples['example_id'] = [examples['example_id'][eidx] for eidx in tokenized_examples['example_idx']]
    
    path_name = output_dir + f"tokenized-stride{stride}-q{max_q_char_len}-seq{max_seq_length}/{split_type}/"
    if not os.path.exists(path_name):
       os.makedirs(path_name)
    # tokenized_examples.(path_name + f'subset{split}')
    
    import json
    # with open(path_name + f'subset{split}.json', 'w') as zipfile:
    #     json.dump(dict(tokenized_examples.data), zipfile)

    import gzip
    with gzip.open(path_name + f'subset{split}.json.gz', 'wt', encoding="utf-8") as zipfile:
       json.dump(dict(tokenized_examples.data), zipfile)
