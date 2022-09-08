
import torch
import json
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from nltk import PorterStemmer, word_tokenize
from rouge import Rouge
# from spacy.lang.en import English
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, DistributedSampler
from tqdm import tqdm
from copy import deepcopy
import os
from os import path, makedirs
import logging

from tensorboardX import SummaryWriter
import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=4, help="Per GPU batch size while training")
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=4, help="Per GPU batch size while evaluation")
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
    parser.add_argument("--top_p", default=0.90, type=float, help="Nucleus size for sampling")
    parser.add_argument("--top_k", default=10, type=int, help="Top k tokens to use")
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

    parser.add_argument("--input_path", type=str, help="Path to train json file")
    parser.add_argument("--output_path", type=str, help="Path to train json file")

    return parser

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
    source_batch = [s for s, _, _, _, _, _ in input_list]
    target_batch = [t for _, t, _, _, _, _ in input_list]
    answer_batch = [a for _, _, a, _, _, _ in input_list]
    item_index = [i for _, _, _, i, _, _ in input_list]
    entity_index = [i for _, _, _, _, i, _ in input_list]
    indices = [i for _, _, _, _, _, i in input_list]

    indices = torch.LongTensor(indices).to(device)

    source_toks = tokenizer.__call__(source_batch, answer_batch, max_length=max_input_length, truncation=True, pad_to_max_length=True)
    source_ids, source_mask = (
        torch.LongTensor(source_toks["input_ids"]).to(device),
        torch.LongTensor(source_toks["attention_mask"]).to(device),
    )

    target_toks = tokenizer.__call__(target_batch, max_length=max_answer_length, truncation=True, pad_to_max_length=True)
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
        "example_indices": indices,
        "item_index": deepcopy(item_index),
        "entity_index": deepcopy(entity_index)
    }
    return model_inputs

def get_questions(generated_ids, item_ids, entity_ids):
    generated_text = list()
    for answer_ids in generated_ids:
        generated_text.append(tokenizer.decode(answer_ids, skip_special_tokens=False).strip())
    
    generated_questions = dict()
    for index, text in enumerate(generated_text):
        item_index = item_ids[index]
        entity_index = entity_ids[index]
        generated_item = dict()
        if "madeupword0000" in text:
            if "madeupword0001" in text:
                generated_item["question"] = text.split("madeupword0001")[1].replace("</s>", "").replace("<pad>", "").strip()
                generated_item["answer"] = text.split("madeupword0000")[1].split("madeupword0001")[0].strip()
               
            elif "madeup word0001" in text:
                generated_item["question"] = text.split("madeup word0001")[1].replace("</s>", "").replace("<pad>", "").strip()
                generated_item["answer"] = text.split("madeupword0000")[1].split("madeup word0001")[0].strip()
                
            elif "madeupsword0001" in text:
                generated_item["question"] = text.split("madeupsword0001")[1].replace("</s>", "").replace("<pad>", "").strip()
                generated_item["answer"] = text.split("madeupword0000")[1].split("madeupsword0001")[0].strip()
                 
            elif "made upword0001" in text:
                generated_item["question"] = text.split("made upword0001")[1].replace("</s>", "").replace("<pad>", "").strip()
                generated_item["answer"] = text.split("madeupword0000")[1].split("made upword0001")[0].strip()
                 
            elif "madeupwords0001" in text:
                generated_item["question"] = text.split("madeupwords0001")[1].replace("</s>", "").replace("<pad>", "").strip()
                generated_item["answer"] = text.split("madeupword0000")[1].split("madeupwords0001")[0].strip()
                
            elif "madeup0001" in text:
                generated_item["question"] = text.split("madeup0001")[1].replace("</s>", "").replace("<pad>", "").strip()
                generated_item["answer"] = text.split("madeupword0000")[1].split("madeup0001")[0].strip()
                
            else:
                generated_item["question"] = ""
        else:
            generated_item["question"] = ""
        if item_index not in generated_questions:
            generated_questions[item_index] = dict()
        if entity_index in generated_questions[item_index]:
            assert False
        generated_questions[item_index][entity_index] = deepcopy(generated_item)
    
    return generated_questions

def get_batches_of_3(input_data):
    batches = int(len(input_data)/5)
    output_data = list()
    for i in range(0, batches):
        output_data.append(deepcopy(input_data[i*5:i*5+5]))
    return output_data

def generate(model, tokenizer, args):

    input_data = json.load(open(args.input_path, "r"))
    
    # PATTERNS:
    # 1) <s>The... parent. madeupword0000 outstanding madeup word0001 what percentage of a corporation's stock is non-controlling</s><pad>
    # 2) <s><P>... Jacobs. madeupword0000 Eagles madeupwords0001 who sings the song love will keep us alive
    # 3) madeupword0001
    # 4) <s>Andrew... N. Y. madeupword0000 escaped Madeupword0001 what happened to the inmates in the ny prison</s><pad>

    n_gpu = torch.cuda.device_count()
    batch_size = args.per_gpu_eval_batch_size * n_gpu

    batch_data = get_batches_of_3(input_data)

    output_data = list()

    for item in tqdm(batch_data):
        try:
            data = list()
            for item_index, batch_item in enumerate(item):
                for entity_index, entity in enumerate(batch_item["generate_entities"]):
                    data.append(["<P> " + batch_item["text"] + " </P>", "", entity["string"], item_index, entity_index, 0]) 

            model_inputs = make_input_batch(data, tokenizer=tokenizer, max_input_length=args.max_length, max_answer_length=args.max_answer_length, device=args.device)
            
            generated_ids = model.generate(
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    max_length=args.max_answer_length,
                    do_sample=True,
                    temperature=1.0,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    eos_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1,
                    decoder_start_token_id=tokenizer.bos_token_id,
                )

            questions = get_questions(generated_ids, model_inputs["item_index"], model_inputs["entity_index"])
            
            for item_index, batch_item in enumerate(item):
                new_item = deepcopy(batch_item)
                generated_questions = list()   
                for entity_index, entity in enumerate(batch_item["generate_entities"]):
                    generated_question = questions[item_index][entity_index]
                    if generated_question["question"] != "":
                        generated_item = deepcopy(entity)
                        generated_item["question"] = generated_question["question"]
                        generated_item["answer"] = generated_question["answer"]
                        generated_questions.append(deepcopy(generated_item))
                new_item["generated_questions"] = deepcopy(generated_questions)  
                output_data.append(deepcopy(new_item))
        except:
            continue        
    
    json.dump(output_data, open(args.output_path, "w"))                    

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
    logging.info("Starting ELI5")

    parser = get_parser()
    s2s_args = parser.parse_args()

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

    model, tokenizer = create_model_and_tokenizer(model_name=s2s_args.model_name, args = s2s_args)

    generate(model=model, tokenizer=tokenizer, args = s2s_args)