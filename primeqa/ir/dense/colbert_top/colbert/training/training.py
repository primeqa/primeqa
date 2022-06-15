import os
import errno
import time
import torch
import math
import random
import torch.nn as nn
import numpy as np
import glob
import sys
import re
import copy
from collections import OrderedDict

from queue import Empty

from primeqa.ir.dense.colbert_top.colbert.infra.run import Run

from transformers import AdamW, get_linear_schedule_with_warmup
from primeqa.ir.dense.colbert_top.colbert.infra import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.training.rerank_batcher import RerankBatcher
from primeqa.ir.dense.colbert_top.colbert.training.eager_batcher_v2 import EagerBatcher  # support text input

from primeqa.ir.dense.colbert_top.colbert.utils.amp import MixedPrecisionManager
from primeqa.ir.dense.colbert_top.colbert.training.lazy_batcher import LazyBatcher
from primeqa.ir.dense.colbert_top.colbert.parameters import DEVICE

from primeqa.ir.dense.colbert_top.colbert.modeling.colbert import ColBERT
from primeqa.ir.dense.colbert_top.colbert.modeling.reranker.electra import ElectraReranker

from primeqa.ir.dense.colbert_top.colbert.utils import signals
from primeqa.ir.dense.colbert_top.colbert.utils.utils import torch_load_dnn
from primeqa.ir.dense.colbert_top.colbert.utils.utils import print_message, save_checkpoint
from primeqa.ir.dense.colbert_top.colbert.training.utils import print_progress, manage_checkpoints_consumed_all_triples, manage_checkpoints_with_path_save


def calculate_distance(student_out, teacher_out):
    '''calculate the distance between student output tokens and teacher output tokens'''
    # start = time.time()
    prod = teacher_out.matmul(student_out.transpose(1, 2))
    student_out_norm = torch.norm(student_out, p=2, dim=-1)
    teacher_out_norm = torch.norm(teacher_out, p=2, dim=-1)
    m = teacher_out_norm.unsqueeze(2) * student_out_norm.unsqueeze(1)
    esp = torch.ones_like(m) * 10**-8
    distance = torch.ones_like(m) - prod /(m + esp)
    # end = time.time()
    # print("time to calculation distance matrix: ", end - start)
    return distance

def align(maxlen, student_out, teacher_out, teacher_queries):
    '''re-order teacher output tokens so that it aligns with
    student output tokens with greedy search'''
    batch_distance_array=calculate_distance(student_out, teacher_out)
    batch_distance_array = batch_distance_array.cpu().detach().numpy()
    for idx, distance_array in enumerate(batch_distance_array):
        swaps = []
        for i in range(maxlen):
            minValue =np.amin(distance_array)
            indexs = np.where(distance_array == np.amin(minValue))
            #get the index of the first min value
            i, j = indexs[0][0], indexs[1][0]
            #swap arrary row i and row j
            distance_array[[i, j]] = distance_array[[j, i]]
            distance_array[j, :] = 10  #anything larger than 1 to avoid double count
            distance_array[:, j] = 10
            swaps.append((i,j))
        for swap in swaps:
            teacher_out[idx][[swap[0], swap[1]]] = teacher_out[idx][[swap[1], swap[0]]]
            teacher_queries[0][idx][[swap[0], swap[1]]] = teacher_queries[0][idx][[swap[1], swap[0]]]

def train(config: ColBERTConfig, triples, queries=None, collection=None):

    if config.rank < 1:
        config.help()

    assert not ( config.use_ib_negatives and config.distill_query_passage_separately ) , f" Simultaneous use of --use_ib_negatives and --distill_query_passage_separately options is not supported (yet)"

    # When checkpoint specified, we need to get model_type from previous run if necessary or as a model type
    if config.checkpoint is not None:
        if config.checkpoint.endswith('.dnn') or config.checkpoint.endswith('.model'):
            # adding "or config.checkpoint.endswith('.model')" to be compatible with V1
            checkpoint = torch_load_dnn(config.checkpoint)
            # if checkpoint['model_type'] is not None:
            assert 'model_type' in checkpoint and checkpoint['model_type'] is not None, f"missing or invalid  checkpoint type in {config.checkpoint}"
            config.model_type = checkpoint['model_type']

        # Use checkpoint as a model type
        elif config.checkpoint == 'bert-base-uncased' or config.checkpoint =='bert-large-uncased' \
                or config.checkpoint == 'xlm-roberta-base' or config.checkpoint == 'xlm-roberta-large':
            config.model_type = config.checkpoint
        else:
            print_message(f"unsupported checkpoint type or format: {config.checkpoint}")
            raise NotImplementedError

    print_message(f"model type: {config.model_type}")

    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)
    torch.cuda.manual_seed_all(12345)

    assert config.bsize % config.nranks == 0, (config.bsize, config.nranks)
    config.bsize = config.bsize // config.nranks

    print_message("Using config.bsize =", config.bsize, "(per process) and config.accumsteps =", config.accumsteps)

    # the reader , the proper tokenizer is based on model type
    if collection is not None:
        if config.reranker:
            reader = RerankBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        else:
            reader = LazyBatcher(config, triples, queries, collection, (0 if config.rank == -1 else config.rank), config.nranks)
        assert config.teacher_checkpoint is None, "Student/Teacher training is not supported for numerical triples (yet)"
    else:
        # support text input
        reader = EagerBatcher(config, triples, (0 if config.rank == -1 else config.rank), config.nranks)
        if config.teacher_checkpoint is not None:
            teacher_reader = EagerBatcher(config, config.teacher_triples, (0 if config.rank == -1 else config.rank), config.nranks)

    if not config.reranker:
        colbert = ColBERT(name=config.model_type, colbert_config=config)

        # add support pre-trained representation
        if config.init_from_lm is not None and config.checkpoint is None:
            # checkpoint should override init_from_lm since it continues an already init'd run
            print_message(f"#> Load init from lm {config.init_from_lm}")
            if DEVICE == torch.device("cuda"):
                lmweights = torch.load(config.init_from_lm)
            else:    # expect path to pytorch_model.bin
                lmweights = torch.load(config.init_from_lm, map_location=torch.device('cpu'))  # expect path to pytorch_model.bin


            lmweights['model.linear.weight'] = colbert.linear.weight
            # we don't need the keys in the lm head
            keys_to_drop = ['lm_head.dense.weight', 'lm_head.dense.bias', 'lm_head.layer_norm.weight',
                            'lm_head.layer_norm.bias', 'lm_head.decoder.weight', 'lm_head.decoder.bias', 'lm_head.bias']
            if config.model_type == 'xlm-roberta-base':
                # TODO other model types may have a few extra keys to handle also ...

                # resolve conflict between bert and roberta
                lmweights_new = OrderedDict([(re.sub(r'^roberta\.', 'model.bert.', key), value) for key, value in lmweights.items()])

                lmweights_new['model.bert.pooler.dense.weight'] = colbert.bert.pooler.dense.weight
                lmweights_new['model.bert.pooler.dense.bias'] = colbert.bert.pooler.dense.bias

                # I don't know what roberta.embeddings.position_ids is but it doesn't seem to be part of the model ...
                # keys_to_drop += ['roberta.embeddings.position_ids']
            elif config.model_type == 'tinybert':
                keys_to_drop = ["cls.predictions.bias", "cls.predictions.transform.dense.weight",
                                "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight",
                                "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight",
                                "cls.seq_relationship.weight", "cls.seq_relationship.bias", "fit_denses.0.weight",
                                "fit_denses.0.bias", "fit_denses.1.weight", "fit_denses.1.bias", "fit_denses.2.weight",
                                "fit_denses.2.bias", "fit_denses.3.weight", "fit_denses.3.bias", "fit_denses.4.weight",
                                "fit_denses.4.bias"]

            for k in keys_to_drop:
                lmweights_new.pop(k)

            colbert.load_state_dict(lmweights_new,False)

        # load from checkpoint if checkpoint is an actual model
        if config.checkpoint is not None:
            if config.checkpoint.endswith('.dnn') or config.checkpoint.endswith('.model'):
                print_message(f"#> Starting from checkpoint {config.checkpoint}")
                checkpoint = torch.load(config.checkpoint, map_location='cpu')

                try:
                    colbert.load_state_dict(checkpoint['model_state_dict'])
                except:
                    print_message("[WARNING] Loading checkpoint with strict=False")
                    colbert.load_state_dict(checkpoint['model_state_dict'], strict=False)

        if config.teacher_checkpoint is not None:
            teacher_colbert = ColBERT(name=config.teacher_model_type, colbert_config=config)

            if config.teacher_checkpoint.endswith('.dnn') or config.teacher_checkpoint.endswith('.model'):
                print_message(f"#> Loading teacher checkpoint {config.teacher_checkpoint}")
                teacher_checkpoint = torch.load(config.teacher_checkpoint, map_location='cpu')

                try:
                    teacher_colbert.load_state_dict(teacher_checkpoint['model_state_dict'])
                except:
                    print_message("[WARNING] Loading checkpoint with strict=False")
                    teacher_colbert.load_state_dict(teacher_checkpoint['model_state_dict'], strict=False)
    else:
        colbert = ElectraReranker.from_pretrained(config.checkpoint)


    colbert = colbert.to(DEVICE)
    colbert.train()

    if config.teacher_checkpoint is not None:
        teacher_colbert = teacher_colbert.to(DEVICE)
        if config.distill_query_passage_separately:
            #assert False, "distill_query_passage_separately functionality is not supported (yet)"
            print_message("distill_query_passage_separately functionality is not supported (yet)")
            if config.loss_function == 'MSE':
                student_teacher_loss_fct = torch.nn.MSELoss()
            else:
                student_teacher_loss_fct = torch.nn.L1Loss()
        else:
            student_teacher_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")

    if DEVICE == torch.device("cuda"):
        colbert = torch.nn.parallel.DistributedDataParallel(colbert, device_ids=[config.rank],
                                                        output_device=config.rank,
                                                        find_unused_parameters=True)
        if config.teacher_checkpoint is not None:
            teacher_colbert = torch.nn.parallel.DistributedDataParallel(teacher_colbert, device_ids=[config.rank],
                                                output_device=config.rank,
                                                find_unused_parameters=True)


    optimizer = AdamW(filter(lambda p: p.requires_grad, colbert.parameters()), lr=config.lr, eps=1e-8)
    optimizer.zero_grad()

    if config.resume_optimizer:
        print_message(f"#> Resuming optimizer from checkpoint {config.checkpoint}")
        torch.set_rng_state(checkpoint['torch_rng_state'].to(torch.get_rng_state().device))
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([ state.to(torch.cuda.get_rng_state_all()[pos].device) for pos, state in enumerate(checkpoint['torch_cuda_rng_states']) ] )
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler = None
    if config.warmup is not None:
        print_message(f"#> LR will use {config.warmup} warmup steps and linear decay over {config.maxsteps} steps.")
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
         train_loss = checkpoint['train_loss']

         # reader.skip_to_batch(start_batch_idx, checkpoint['arguments']['bsize'])
         reader.skip_to_batch(start_batch_idx, config.bsize)
         if config.teacher_checkpoint is not None:
            teacher_reader.skip_to_batch(start_batch_idx, config.bsize)

    maxsteps = min(config.maxsteps, math.ceil((config.epochs * len(reader)) / (config.bsize * config.nranks)))

    path = os.path.join(Run().path_, 'checkpoints')
    if not os.path.exists(path):
        os.makedirs(path)

    name = os.path.join(path, "colbert-EXIT.dnn")
    # arguments = config.input_arguments.__dict__
    exit_queue = signals.checkpoint_on_exit(config.rank)

    print_message(f"maxsteps: {config.maxsteps}")
    print_message(f"{config.epochs} epochs of {len(reader)} examples")
    print_message(f"batch size: {config.bsize}")
    print_message(f"maxsteps set to {maxsteps}")

    print_message(f"start batch idx: {start_batch_idx}")

    # TODO: unify the student/teacher and student-only cases
    if config.teacher_checkpoint is not None:
        for batch_idx, BatchSteps, teacher_BatchSteps in zip(range(start_batch_idx, maxsteps), reader, teacher_reader):
            if (warmup_bert is not None) and warmup_bert <= batch_idx:
                set_bert_grad(colbert, True)
                warmup_bert = None

            # support shuffle_every_epoch option
            n_instances = batch_idx * config.bsize * config.nranks
            if (n_instances + 1) % len(reader) < config.bsize * config.nranks:
                print_message("#> ====== Epoch {}...".format((n_instances+1) // len(reader)))
                # AttributeError: 'ColBERTConfig' object has no attribute 'shuffle_every_epoch'
                if config.shuffle_every_epoch:
                    print_message("[WARNING] Data shuffling is not supported (yet) for Student/Teacher training")
                else:
                    print_message("#> Shuffling not specified.")

            this_batch_loss = 0.0

            for queries_passages, teacher_queries_passages in zip(BatchSteps, teacher_BatchSteps):
                assert(config.teacher_model_type is not None or torch.equal(queries_passages[1][0], teacher_queries_passages[1][0]))

                with amp.context():
                    if config.distill_query_passage_separately :
                        if config.query_only:
                            assert False, "Training with --query-only option is not supported (yet)."
                        else:
                            queries, passages, target_scores = queries_passages
                            encoding = [queries, passages]
                            scores, student_output_q, student_output_p = colbert(*encoding)

                            with torch.no_grad():
                                teacher_queries, teacher_passages, teacher_target_scores = teacher_queries_passages
                                teacher_encoding = [teacher_queries, teacher_passages]
                                teacher_scores, teacher_output_q, teacher_output_p  = teacher_colbert(*teacher_encoding)

                            teacher_queries_toks_masks = (teacher_queries_passages[0][0].repeat_interleave(config.nway, dim=0).contiguous(), teacher_queries_passages[0][1].repeat_interleave(config.nway, dim=0).contiguous())
                            teacher_queries = copy.deepcopy(teacher_queries_toks_masks)
                            maxlen = config.query_maxlen
                            align(maxlen, student_output_q, teacher_output_q, teacher_queries)
                            loss = config.query_weight * student_teacher_loss_fct(student_output_q, teacher_output_q) + (1 - config.query_weight)*student_teacher_loss_fct(student_output_p, teacher_output_p)
                    else:
                        try:
                            queries, passages, target_scores = queries_passages
                            encoding = [queries, passages]
                        except:
                            encoding, target_scores = queries_passages
                            encoding = [encoding.to(DEVICE)]

                        scores = colbert(*encoding)

                        if config.use_ib_negatives:
                            scores, ib_loss = scores

                        scores = scores.view(-1, config.nway)

                        with torch.no_grad():
                            try:
                                teacher_queries, teacher_passages, teacher_target_scores = teacher_queries_passages
                                teacher_encoding = [teacher_queries, teacher_passages]
                            except:
                                teacher_encoding, teacher_target_scores = teacher_queries_passages
                                teacher_encoding = [teacher_encoding.to(DEVICE)]

                            teacher_scores = teacher_colbert(*teacher_encoding)

                            if config.use_ib_negatives:
                                teacher_scores, teacher_ib_loss = teacher_scores

                            teacher_scores = teacher_scores.view(-1, config.nway)

                        loss = student_teacher_loss_fct(
                                    torch.nn.functional.log_softmax(scores / config.student_teacher_temperature, dim=-1),
                                    torch.nn.functional.softmax(teacher_scores / config.student_teacher_temperature, dim=-1),
                                ) * (config.student_teacher_temperature ** 2)

                    loss = loss / config.accumsteps

                if config.rank < 1:
                    print_progress(scores.view(-1,2) if config.distill_query_passage_separately else scores)

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
                    # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, arguments)
                    save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, config.model_type)
                    # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, config.model_type, arguments)
                    sys.exit(0)
                except Empty:
                    # manage_checkpoints(config, colbert, optimizer, amp, batch_idx + 1, num_per_epoch, epoch_idx, train_loss)
                    manage_checkpoints_with_path_save(config, colbert, optimizer, amp, batch_idx + 1, num_per_epoch, epoch_idx, train_loss)

    else:
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
                    # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, arguments)
                    save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, config.model_type)
                    # save_checkpoint(name, epoch_idx, batch_idx, colbert, optimizer, amp, train_loss, config.model_type, arguments)
                    sys.exit(0)
                except Empty:
                    # manage_checkpoints(config, colbert, optimizer, amp, batch_idx + 1, num_per_epoch, epoch_idx, train_loss)
                    manage_checkpoints_with_path_save(config, colbert, optimizer, amp, batch_idx + 1, num_per_epoch, epoch_idx, train_loss)

    # save last model
    name = os.path.join(path, "colbert-LAST.dnn")
    print_message('name:' + name)
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

    # just return latest file
    return latest_file


def set_bert_grad(colbert, value):
    try:
        for p in colbert.bert.parameters():
            assert p.requires_grad is (not value)
            p.requires_grad = value
    except AttributeError:
        set_bert_grad(colbert.module, value)
