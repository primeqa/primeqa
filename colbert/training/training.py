import os
import errno
import time
import torch
import math
import random
import torch.nn as nn
import numpy as np
import glob
from queue import Empty

from colbert.infra.run import Run

from transformers import AdamW, get_linear_schedule_with_warmup
from colbert.infra import ColBERTConfig
from colbert.training.rerank_batcher import RerankBatcher
from colbert.training.eager_batcher_v2 import EagerBatcher  # support text input

from colbert.utils.amp import MixedPrecisionManager
from colbert.training.lazy_batcher import LazyBatcher
from colbert.parameters import DEVICE

from colbert.modeling.colbert import ColBERT
from colbert.modeling.reranker.electra import ElectraReranker

from colbert.utils import signals
from colbert.utils.utils import torch_load_dnn
from colbert.utils.utils import print_message
from colbert.training.utils import print_progress, manage_checkpoints, manage_checkpoints_consumed_all_triples


def train(config: ColBERTConfig, triples, queries=None, collection=None):
    config.checkpoint = config.checkpoint or 'bert-base-uncased'

    if config.rank < 1:
        # config.help()
        # remove input arguments(dict) to make help happy
        config_without_input_arguments = config
        config_without_input_arguments.input_arguments = None
        config_without_input_arguments.help()

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
    else:
        # support text input
        reader = EagerBatcher(config, triples, (0 if config.rank == -1 else config.rank), config.nranks)

    if not config.reranker:
        if config.checkpoint.endswith('.dnn') or config.checkpoint.endswith('.model'):
            # adding "or config.checkpoint.endswith('.model')" to be compatible with V1
            dnn = torch_load_dnn(config.checkpoint)
            model_type = dnn.get('arguments', {}).get('model', 'bert-base-uncased')
            # "model" should be "model_type" defined at command line in previous run ?
            colbert = ColBERT(name=model_type, colbert_config=config)
        else:
            colbert = ColBERT(name=config.checkpoint, colbert_config=config)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)

    if config.checkpoint.endswith('.dnn') or config.checkpoint.endswith('.model'):
        print_message(f"#> Starting from checkpoint {config.checkpoint}")
        checkpoint = torch.load(config.checkpoint, map_location='cpu')

        try:
            colbert.load_state_dict(checkpoint['model_state_dict'])
        except:
            print_message("[WARNING] Loading checkpoint with strict=False")
            colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)


    colbert = colbert.to(DEVICE)
    colbert.train()

    #  set 0 when debug
    colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)

    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    if config.resume_optimizer:
        print_message(f"#> Resuming optimizer from checkpoint {config.checkpoint}")
        torch.set_rng_state(checkpoint['torch_rng_state'].to(torch.get_rng_state().device))
        torch.cuda.set_rng_state_all([ state.to(torch.cuda.get_rng_state_all()[pos].device) for pos, state in enumerate(checkpoint['torch_cuda_rng_states']) ] )
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if config.warmup is not None:
        print(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup,
                                                    num_training_steps=config.maxsteps)
    
    warmup_bert = config.warmup_bert
    if warmup_bert is not None:
        set_bert_grad(colbert, False)

    amp = MixedPrecisionManager(config.amp)
    if config.resume_optimizer and config.amp:
        amp.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    labels = torch.zeros(config.bsize, dtype=torch.long, device=DEVICE)

    start_time = time.time()
    train_loss = None
    train_loss_mu = 0.999

    start_batch_idx = 0

    if config.resume:
         assert config.checkpoint is not None
         start_batch_idx = checkpoint['batch']

         reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])

    maxsteps = min(config.maxsteps, math.ceil((config.epochs * len(reader)) / (config.bsize * config.nranks)))

    path = os.path.join(Run().path_, 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    name = os.path.join(path, "colbert-EXIT.dnn")
    arguments = config.input_arguments.__dict__
    exit_queue = signals.checkpoint_on_exit(config.rank)

    print_message(f"maxsteps: {config.maxsteps}")
    print_message(f"{config.epochs} epochs of {len(reader)} examples")
    print_message(f"batch size: {config.bsize}")
    print_message(f"maxsteps set to {maxsteps}")

    for batch_idx, BatchSteps in zip(range(start_batch_idx, config.maxsteps), reader):
        if (warmup_bert is not None) and warmup_bert <= batch_idx:
            set_bert_grad(colbert, True)
            warmup_bert = None

        # support shuffle_every_epoch option
        n_instances = batch_idx * config.bsize * config.nranks
        if (n_instances + 1) % len(reader) < config.bsize * config.nranks:
            print_message("#> ====== Epoch {}...".format((n_instances+1) // len(reader)))
            # AttributeError: 'ColBERTConfig' object has no attribute 'shuffle_every_epoch'
            if config.shuffle_every_epoch:
                print_message("#> Shuffling ...")
                reader.shuffle()
            else:
                print_message("#> Shuffling not specified.")


        this_batch_loss = 0.0

        for batch in BatchSteps:
            with amp.context():
                try:
                    queries, passages, target_scores = batch
                    encoding = [queries, passages]
                except:
                    encoding, target_scores = batch
                    encoding = [encoding.to(DEVICE)]

                scores = colbert(*encoding)

                if config.use_ib_negatives:
                    scores, ib_loss = scores

                scores = scores.view(-1, config.nway)

                if len(target_scores) and not config.ignore_scores:
                    target_scores = torch.tensor(target_scores).view(-1, config.nway).to(DEVICE)
                    target_scores = target_scores * config.distillation_alpha
                    target_scores = torch.nn.functional.log_softmax(target_scores, dim=-1)

                    log_scores = torch.nn.functional.log_softmax(scores, dim=-1)
                    loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)(log_scores, target_scores)
                else:
                    loss = nn.CrossEntropyLoss()(scores, labels[:scores.size(0)])

                if config.use_ib_negatives:
                    if config.rank < 1:
                        print('\t\t\t\t', loss.item(), ib_loss.item())

                    loss += ib_loss

                loss = loss / config.accumsteps

            if config.rank < 1:
                print_progress(scores)

            amp.backward(loss)

            this_batch_loss += loss.item()

        train_loss = this_batch_loss if train_loss is None else train_loss
        train_loss = train_loss_mu * train_loss + (1 - train_loss_mu) * this_batch_loss

        amp.step(colbert, optimizer, scheduler)

        if config.rank < 1:
            print_message(batch_idx, train_loss)

            num_per_epoch = len(reader)
            epoch_idx = ((batch_idx + 1) * config.bsize * config.nranks) // num_per_epoch - 1
            try:
                exit_queue.get_nowait()
                save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, arguments)
                sys.exit(0)
            except Empty:
                manage_checkpoints(config, colbert, optimizer, amp, batch_idx + 1, num_per_epoch, epoch_idx)

    # save last model
    name = os.path.join(path, "colbert-LAST.dnn")
    print('name:' + name)
    list_of_files = glob.glob(f'{path}/*.model')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    # Run.info(f"Make a sym link of {latest_file} to {name}")
    print_message(f"Make a sym link of {latest_file} to {name}")
    try:
        os.symlink(latest_file, name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(name)
            os.symlink(latest_file, name)
        else:
            raise

    if config.rank < 1:
        print_message("#> Done with all triples!")
        ckpt_path = manage_checkpoints_consumed_all_triples(config, colbert, optimizer, batch_idx+1, savepath=None, consumed_all_triples=True)
        return ckpt_path  # TODO: This should validate and return the best checkpoint, not just the last one.


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
