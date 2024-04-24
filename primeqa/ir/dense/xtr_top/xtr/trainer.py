import os
import re
import sys
import time
import math
import glob
import json
import copy
import random
import numpy as np

import torch
import torch.nn as nn

from transformers import AutoConfig
from transformers import AdamW, get_linear_schedule_with_warmup

from primeqa.ir.dense.xtr_top.xtr.modeling.XTR import XTR
from primeqa.ir.dense.xtr_top.xtr.tokenization.eager_batcher import EagerBatcher  # support text input


def train(config):
    random.seed(config.rng_seed)
    np.random.seed(config.rng_seed)
    torch.manual_seed(config.rng_seed)
    torch.cuda.manual_seed_all(config.rng_seed)

    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print(f"Using config.bsize = {config.bsize} (per process) and config.accumsteps = {config.accumsteps}")

    # support text input
    reader = EagerBatcher(config, config.triples, (0 if config.rank == -1 else config.rank), config.nranks)
    tokenizer = reader.tokenizer.tok

    localtime = time.localtime()
    date = f"{localtime.tm_mday}-{localtime.tm_mon}-{localtime.tm_year}"
    output_directory = f"{config.experiment_path}_{date}"
    os.makedirs(output_directory, exist_ok=True)

    arguments = dict(config._get_kwargs())
    if 'input_arguments' in arguments:
        del arguments['input_arguments']

    with open(f"{output_directory}/args.json", "w") as fp:
        json.dump([arguments], fp, indent=4)

    model_config = AutoConfig.from_pretrained(config.model_name_or_path)
    model_config.dim = config.dim
    xtr_model = XTR.from_pretrained(config.model_name_or_path, config=model_config)
        
    xtr_model = xtr_model.to(DEVICE)
    xtr_model.train()

    optimizer = AdamW(filter(lambda p: p.requires_grad, xtr_model.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    maxsteps = min(config.maxsteps, math.ceil((config.epochs * len(reader)) / (config.bsize * config.nranks)))

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=maxsteps)
    
    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    print(f"{config.epochs} epochs of {len(reader)} examples")
    print(f"batch size: {config.bsize}")
    print(f"maxsteps set to {maxsteps}")

    print(f"start batch idx: {start_batch_idx}")

    batch_loss = 0.0
    batch_accuracy = 0.0
    for batch_idx, BatchSteps in zip(range(start_batch_idx, maxsteps), reader):
        # support shuffle_every_epoch option
        n_instances = batch_idx * config.bsize * config.nranks
        if (n_instances + 1) % len(reader) < config.bsize * config.nranks:
            print(f"#> ====== Epoch {(n_instances+1) // len(reader)}")
            if config.shuffle_every_epoch:
                print("#> Shuffling ...")
                reader.shuffle()
            else:
                print("#> Shuffling not specified.")

        for bid, batch in enumerate(BatchSteps, 1):
            query_ids, query_attention_mask = [qt.to(DEVICE) for qt in batch[0][:2]]
            doc_ids, doc_attention_mask = [dt.to(DEVICE) for dt in batch[1][:2]]

            loss, accuracy = xtr_model(query_ids, doc_ids, 
                                    query_attention_mask, doc_attention_mask,
                                    nway=config.nway, k=config.k_train)
    
            #loss = loss / config.accumsteps
            assert loss.requires_grad == True

            loss.backward()
            batch_loss += loss.item()
            batch_accuracy += accuracy.item()

        train_loss = batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * batch_loss

        if (batch_idx + 1) % 50 == 0:
            print(f"Batch ID: {batch_idx+1}, Train Loss: {batch_loss/50}, Batch Accuracy: {batch_accuracy/50}")
            batch_loss = 0.0
            batch_accuracy = 0.0

        torch.nn.utils.clip_grad_norm_(xtr_model.parameters(), config.max_grad_clip)

        optimizer.step()
        
        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad()

        model_to_save = (
                       xtr_model.module if hasattr(xtr_model, "module") else xtr_model
                       )  # Take care of distributed/parallel training

        if (batch_idx+1) % config.save_steps == 0:
            save_path = f"{output_directory}/xtr-batch_{batch_idx+1}"
            tokenizer.save_pretrained(save_path)
            model_to_save.config.train_loss = train_loss  
            model_to_save.save_pretrained(save_path)
