import functools
from time import time
import torch
import json
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from math import ceil
import logging
from nltk import PorterStemmer, word_tokenize
from rouge import Rouge
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from copy import deepcopy
import os
from os import path, makedirs

from tensorboardX import SummaryWriter
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=4, help="Per GPU batch size while training")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=2, help="Per GPU batch size while evaluation")
    parser.add_argument("--train_batch_size", type=int, default=128, help="Total batch size while training")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum input length")
    parser.add_argument("--max_answer_length", type=int, default=360, help="Maximum answer length while training")

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        help="The output directory where the model checkpoints and predictions will be written.",
    )
    parser.add_argument("--model_save_name", type=str, default='pytorch_model_epoch_%d', help="Name for folder to save model")
    parser.add_argument("--model_name", type=str, default='facebook/bart-large', help="Language model to load")

    parser.add_argument("--learning_rate", default=2e-5, type=float, help="The initial learning rate for Adam")
    parser.add_argument("--num_epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="Beta_1 for Adam optimizer.")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help='Beta_2 for Adam optimizer')
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for"
                             " Bert Adam. E.g., 0.1=10%% of training.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")

    parser.add_argument("--min_generate_len", type=int, default=128, help="Minimum answer length while generation")
    parser.add_argument("--max_generate_len", type=int, default=384, help="Maximum answer length while generation")
    parser.add_argument("--beam_size", type=int, default=8, help="Beam size while generation")
    parser.add_argument("--logging_steps", type=int, default=10, help="random seed for initialization")

    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--train_file", type=str, help="Path to train json file")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--predict_file",  type=str, help="Path to dev json file")
    parser.add_argument("--use_distributed_cache", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument('--model_weights', default=None, type=str,
                        required=False,
                        help="Either a filepath to a previously trained Bert QA model weights (i.e."
                             " the state_dict) OR model directory")

    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")

    parser.add_argument("--no_support", action='store_true', help="Whether to use support document.")
    parser.add_argument("--do_sample", action='store_true', help="Whether to sample while generation")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 "
                             "set to True. 0 (default value): dynamic loss scaling. Positive power "
                             "of 2: static loss scaling value.")

    return parser

def get_arguments():
    
    parser = get_parser()
    args = parser.parse_args()

    assert args.output_dir

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. ".format(
                args.output_dir
            )
        )

    if args.fp16:
        try:
            from apex import amp
            # HACK: dirty trick to import amp once so that it's accessible outside this method
            globals()['amp'] = amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    return args

def create_model_and_tokenizer(model_name, args):

    device = args.device
    from_file = args.model_weights
    local_rank = args.local_rank

    language_model_name = model_name
    logging.info('***** Loading Model %s *****\n' % language_model_name)
    logging.info('  device = %s' % device)
    if args.use_distributed_cache:
        model_args_dict = {'cache_dir': path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(local_rank))}
    else:
        model_args_dict = {'cache_dir': PYTORCH_PRETRAINED_BERT_CACHE}
    if from_file is None:
        logging.info(
            '  Loading default language model pretraining weights: %s' % language_model_name)
        logging.info(model_args_dict)
    elif path.isdir(from_file):
        logging.info(
            '  Loading pre-trained model from directory: %s' % from_file)
        language_model_name = from_file
    elif path.isfile(from_file):
        logging.info('  Loading default language model %s and then replacing weights with %s ' %
                     (language_model_name, from_file))
        model_args_dict['state_dict'] = torch.load(
            from_file) if str(device) != "cpu" \
            else torch.load(custom_pretraining_model_weights, map_location="cpu")
    else:
        raise ValueError('Invalid value for pretrained model weights. It must be one of '
                         '(None, model dir, model state weights dictionary file, but found: %s' %
                         from_file)

    model = BartForConditionalGeneration.from_pretrained(language_model_name, **model_args_dict)

    tokenizer = BartTokenizer.from_pretrained(model_name)

    model.to(device)

    return model, tokenizer

def make_input_batch(input_list, tokenizer, max_input_length, max_answer_length, device):
    source_batch = [s for s, _, _ in input_list]
    target_batch = [t for _, t, _ in input_list]
    indices = [i for _, _, i in input_list]

    indices = torch.LongTensor(indices).to(device)

    source_toks = tokenizer.batch_encode_plus(source_batch,  max_length=max_input_length, truncation=True, pad_to_max_length=True)
    source_ids, source_mask = (
        torch.LongTensor(source_toks["input_ids"]).to(device),
        torch.LongTensor(source_toks["attention_mask"]).to(device),
    )

    target_toks = tokenizer.batch_encode_plus(target_batch, max_length=max_answer_length, truncation=True, pad_to_max_length=True)
    target_ids, target_mask = (
        torch.LongTensor(target_toks["input_ids"]).to(device),
        torch.LongTensor(target_toks["attention_mask"]).to(device),
    )
    labels = target_ids[:, 1:].contiguous().clone()
    labels[target_mask[:, 1:].contiguous() == 0] = -100
    model_inputs = {
        "input_ids": source_ids,
        "attention_mask": source_mask,
        "decoder_input_ids": target_ids[:, :-1].contiguous(),
        "labels": labels,
        "example_indices": indices
    }
    return model_inputs

class SynDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples
    
    def __len__(self):
        return len(self.examples)

    def make_example(self, idx):
        example = self.examples[idx]
        
        return (example["source"], example["target"], idx)


    def __getitem__(self, idx):
        return self.make_example(idx)

def get_optimizer_and_scheduler(model, train_dataset, args):

    num_features = len(train_dataset) 
    logging.info("Num of features: %d" % num_features)

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1

    mini_batch_size = args.per_gpu_train_batch_size * args.n_gpu 

    final_mini_batch_size = mini_batch_size * world_size

    grad_accumulation_steps = args.train_batch_size // final_mini_batch_size

    logging.info("Mini Batch size: %d" % mini_batch_size)
    logging.info("Gradient accumulation steps: %d" % grad_accumulation_steps)

    num_steps_per_epoch = num_features // final_mini_batch_size
    
    logging.info("Num steps per epoch: %d" % num_steps_per_epoch)

    num_update_steps_per_epoch = num_features // args.train_batch_size

    logging.info("Num train steps per epoch: %d" % num_update_steps_per_epoch)

    num_train_steps = num_update_steps_per_epoch * args.num_epochs

    parameters_to_optimize = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in parameters_to_optimize if
                        not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in parameters_to_optimize if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
    ]
    # Disable bias correction like the TF BERT implementation
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate,
                        eps=args.adam_epsilon, betas=(args.adam_beta1, args.adam_beta2), correct_bias=False)

    logging.info('Initialized optimizer: %s' % optimizer)

    warmup_steps = ceil(args.warmup_proportion * num_train_steps)
    t_total = torch.tensor(num_train_steps, dtype=torch.float)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

    logging.info('Initialized learning rate scheduler: %s' % scheduler)

    return optimizer, scheduler

def evaluate(model, tokenizer, dev_dataset, args):
    model.eval()
    num_features = len(dev_dataset)
    logging.info("Num of features: %d" % num_features)
    batch_size = args.per_gpu_eval_batch_size * args.n_gpu
    logging.info("Eval batch size: %d " % batch_size)
    eval_sampler = SequentialSampler(dev_dataset)
    model_collate_fn = functools.partial(
        make_input_batch, tokenizer=tokenizer, max_input_length=args.max_length, max_answer_length=args.max_answer_length, device=args.device)
    data_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=eval_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=False)

    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            loss = model(**batch_inputs)[0]
            if args.n_gpu > 1:
                loss = loss.mean()
            
            loc_loss += loss.item()
            loc_steps += 1

    logging.info("Total eval loss \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))


def compute_rouge(predicted, reference):
    compare_list = [(p, g) for p, g in zip(reference, predicted)]
    stemmer = PorterStemmer()
    rouge = Rouge()
    # tokenizer = English().Defaults.create_tokenizer()
    tokenizer = word_tokenize

    preds = [" ".join([stemmer.stem(str(w)) for w in tokenizer(pred)]) for gold, pred in compare_list]
    golds = [" ".join([stemmer.stem(str(w)) for w in tokenizer(gold)]) for gold, pred in compare_list]
    scores = rouge.get_scores(preds, golds, avg=True)

    logging.info("Rouge 1: %.4f " % scores['rouge-1']['f'])
    logging.info("Rouge 2: %.4f " % scores['rouge-2']['f']) 
    logging.info("Rouge L: %.4f " % scores['rouge-l']['f']) 

# This is the code for standard single GPU decode

def generate(model, tokenizer, dev_dataset, args):
    num_features = len(dev_dataset)
    logging.info("Num of features: %d" % num_features)
    n_gpu = torch.cuda.device_count()
    batch_size = args.per_gpu_eval_batch_size * n_gpu
    
    eval_sampler = SequentialSampler(dev_dataset)
    model_collate_fn = functools.partial(
        make_input_batch, tokenizer=tokenizer, max_input_length=args.max_length, max_answer_length=args.max_answer_length, device=args.device)
    data_loader = DataLoader(dev_dataset, batch_size=batch_size, sampler=eval_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=False)

    generated_answers = list()

    for model_inputs in epoch_iterator:
        generated_ids = model.generate(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            min_length=args.min_generate_len,
            max_length=args.max_generate_len,
            do_sample=False,
            early_stopping=True,
            num_beams=args.beam_size,
            temperature=1.0,
            top_k=None,
            top_p=None,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            num_return_sequences=1,
            decoder_start_token_id=tokenizer.bos_token_id,
        )
        for ans_ids in generated_ids:
            generated_answers.append(tokenizer.decode(ans_ids, skip_special_tokens=True).strip())
    
    gold_answers = dev_dataset.get_answer_strings()
    assert len(gold_answers) == len(generated_answers)
    # compute_rouge(predicted = generated_answers, reference = gold_answers)
    return gold_answers, generated_answers

def save_model(model, args, epoch):
    output_dir = path.join(args.output_dir, args.model_save_name % epoch)
    logging.info('Saving model %s to output dir: %s' % (model, output_dir))
    makedirs(output_dir, exist_ok=True)

    # This is to handle the case when DataParallel or DistributedDataParallel has been applied over the model
    model_to_save = model.module if hasattr(model, "module") else model

    model_to_save.save_pretrained(output_dir)

def train(model, tokenizer, train_dataset, dev_dataset, optimizer, scheduler, args):        

    world_size = torch.distributed.get_world_size() if args.local_rank != -1 else 1
    mini_batch_size = args.per_gpu_train_batch_size * args.n_gpu
    grad_accumulation_steps = args.train_batch_size // ( mini_batch_size * world_size )

    model_collate_fn = functools.partial(
        make_input_batch, tokenizer=tokenizer, max_input_length=args.max_length, max_answer_length=args.max_answer_length, device=args.device)

    global_step = 1

    if args.is_master_node:
        tb_writer = SummaryWriter(logdir=path.join(args.output_dir, 'tensorboard'))

    logging.info('Writing Tensorboard file to directory %s' % path.join(args.output_dir, 'tensorboard'))

    tr_loss, logging_loss = 0.0, 0.0

    for i in tqdm(range(0, args.num_epochs), desc="Running epoch", disable=not args.is_master_node):

        model.train()
        model.zero_grad()

        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
        data_loader = DataLoader(train_dataset, batch_size=mini_batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
        epoch_iterator = tqdm(data_loader, desc="Iteration", disable=not args.is_master_node)

        for step, batch_inputs in enumerate(epoch_iterator):

            outputs = model(**batch_inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if grad_accumulation_steps > 1:
                loss = loss / grad_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    # Retain graph to do forward multiple times for adv
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % grad_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer),
                                                    args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            # some printing within the epoch            
                if args.is_master_node:
                    if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                        tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                        logging_loss = tr_loss

        if not args.is_master_node:
            torch.distributed.barrier()

        if args.is_master_node:
            save_model(model, args, i)

        if args.distributed_training and args.is_master_node:
            torch.distributed.barrier()
    
    if args.is_master_node:
        tb_writer.close()

def set_random_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    logging.info('Set random seed to %s' % seed)

def setup_cuda_device(local_rank):
    distributed_training = local_rank != -1

    if not distributed_training:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    is_master_node = not distributed_training or torch.distributed.get_rank() == 0

    return device, n_gpu, is_master_node, distributed_training


def apply_fp16_and_distributed_gpu_settings(model, optimizer, args):

    distributed_training = args.distributed_training
    fp16 = args.fp16
    fp16_opt_level = args.fp16_opt_level
    n_gpu = args.n_gpu
    local_rank = args.local_rank
    loss_scale = args.loss_scale

    logging.info('  Distributed training = %s' % distributed_training)
    logging.info('  fp16 = %s' % fp16)
    logging.info('  fp16_opt_level = %s' % fp16_opt_level)
    logging.info('  local rank = %s' % local_rank)
    logging.info('  num GPUs = %s' % n_gpu)
    logging.info('  loss_scale = %s' % loss_scale)

    if fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        if loss_scale == 0:
            logging.info('Using dynamic loss scaling')
            loss_scale = "dynamic"
        if optimizer is not None:
            model, optimizer = amp.initialize(model, optimizer, opt_level=fp16_opt_level,
                                              loss_scale=loss_scale)
        else:
            model = amp.initialize(model, opt_level=fp16_opt_level, loss_scale=loss_scale)

    # multi-gpu training (should be after apex fp16 initialization)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if distributed_training:
        try:
            from apex.parallel import DistributedDataParallel as DDP

            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              output_device=local_rank,
                                                              find_unused_parameters=True)

        except ImportError as ex:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and"
                " fp16 training.") from ex

    return model, optimizer

if __name__ == "__main__":

    s2s_args = get_arguments()

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    s2s_args.device, s2s_args.n_gpu , s2s_args.is_master_node, s2s_args.distributed_training = setup_cuda_device(s2s_args.local_rank)

    set_random_seed(s2s_args.seed, s2s_args.n_gpu)    

    logging.info(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" % (
        s2s_args.local_rank,
        s2s_args.device,
        s2s_args.n_gpu,
        bool(s2s_args.distributed_training),
        s2s_args.fp16)
    )

    syn_train = json.load(open(s2s_args.train_file, "r"))
    syn_valid = json.load(open(s2s_args.predict_file, "r"))

    train_dataset = SynDataset(examples=syn_train)
    dev_dataset = SynDataset(examples=syn_valid)

    # Load pretrained model and tokenizer
    if not s2s_args.is_master_node:
        # Make sure only the first process in distributed training will download model & vocab
        torch.distributed.barrier()

    model, tokenizer = create_model_and_tokenizer(model_name=s2s_args.model_name, args = s2s_args)

    if s2s_args.distributed_training and s2s_args.is_master_node:
        torch.distributed.barrier() 

    optimizer, scheduler = get_optimizer_and_scheduler(model = model, train_dataset = train_dataset, args = s2s_args)

    model, optimizer = apply_fp16_and_distributed_gpu_settings(model = model, optimizer = optimizer, args = s2s_args)

    if s2s_args.do_train:
        logging.info("Starting Training")
        train(model = model, tokenizer = tokenizer, train_dataset = train_dataset, dev_dataset = dev_dataset, optimizer = optimizer, scheduler = scheduler, args = s2s_args)
        logging.info("Done Training")
    
    if s2s_args.do_predict:
        logging.info("Starting Evaluation")
        evaluate(model=model, tokenizer=tokenizer, dev_dataset=dev_dataset, args=s2s_args)