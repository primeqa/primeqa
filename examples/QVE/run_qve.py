# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for question-answering on SQuAD."""

import warnings
import argparse
import logging
import os
import random
from torch.distributions.binomial import Binomial
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from tqdm import tqdm, trange
import copy
import json

from transformers import (
    MODEL_FOR_QUESTION_ANSWERING_MAPPING,
    AdamW,
    AutoConfig,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from squad_processing import squad_convert_examples_to_features
from transformers.data.processors.squad import SquadV1Processor, SquadExample
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from model import EXTRACTIVE_HEAD, BertForSequenceClassification

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

PT_LR_SCHEDULER_WARNING = "Please also save or load the state of the optimzer when saving or loading the scheduler."


def reissue_pt_warnings(caught_warnings):
    # Reissue warnings that are not the PT_LR_SCHEDULER_WARNING
    if len(caught_warnings) > 1:
        for w in caught_warnings:
            if w.category != UserWarning or w.message != PT_LR_SCHEDULER_WARNING:
                warnings.warn(w.message, w.category)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def calculate_marginal_info(qa_model_marginal, train_dataset, args):
    # calculate marginal information
    if args.n_gpu > 0:
        qa_model_marginal=torch.nn.DataParallel(qa_model_marginal)
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.dev_batch_size)
    qa_model_marginal.eval()
    marginal_info_start=[]
    marginal_info_end=[]

    for train_batch in tqdm(train_dataloader):
        train_batch = tuple(t.to(args.device) for t in train_batch)
        with torch.no_grad():
            inputs = {
                "input_ids": train_batch[0],
                "attention_mask": train_batch[1],
                "token_type_ids": train_batch[2],
            }

            outputs = qa_model_marginal(**inputs)
            pred_start = torch.sigmoid(outputs[0])
            pred_end = torch.sigmoid(outputs[1])
            true_start = torch.nn.functional.one_hot(train_batch[4], pred_start.shape[1])
            true_end = torch.nn.functional.one_hot(train_batch[5],pred_end.shape[1])
            pred_start_prob = (true_start * pred_start).sum(dim=1)
            pred_end_prob = (true_end * pred_end).sum(dim=1)
            diff_start = torch.abs(1 - pred_start_prob)
            diff_end = torch.abs(1 - pred_end_prob)
            marginal_info_start.append(diff_start)
            marginal_info_end.append(diff_end)

    marginal_info_start = torch.cat(marginal_info_start).view(-1,1)
    marginal_info_end = torch.cat(marginal_info_end).view(-1,1)
    marginal_info=torch.cat((marginal_info_start,marginal_info_end),dim=1)

    return marginal_info


def cal_reward_func(args, dev_dataset, qa_model, type="loss"):
    qa_model.eval()
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # dev_sampler = RandomSampler(dev_dataset,replacement=True,num_samples=args.qve_eval_data_num)
    dev_sampler = SequentialSampler(dev_dataset)
    dev_dataloader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=args.dev_batch_size)
    total_reward = 0
    for dev_batch in dev_dataloader:
        dev_batch = tuple(t.to(args.device) for t in dev_batch)
        with torch.no_grad():
            if type == 'loss':
                inputs = {
                    "input_ids": dev_batch[0],
                    "attention_mask": dev_batch[1],
                    "token_type_ids": dev_batch[2],
                    "start_positions": dev_batch[4],
                    "end_positions": dev_batch[5],
                }

                outputs = qa_model(**inputs)
                loss = outputs[0]
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training
                total_reward -= loss*10

            elif type == 'exact':
                inputs = {
                    "input_ids": dev_batch[0],
                    "attention_mask": dev_batch[1],
                    "token_type_ids": dev_batch[2],
                }
                outputs = qa_model(**inputs)
                pred_start_indices = torch.argmax(outputs[0], dim=1)
                pred_end_indices = torch.argmax(outputs[1], dim=1)
                start_acc = pred_start_indices == dev_batch[4]
                end_acc = pred_end_indices == dev_batch[5]
                acc_num = torch.logical_and(start_acc, end_acc).sum()
                total_reward += acc_num
            elif type == "f1":

                def f1_score(prediction, ground_truth):
                    prediction=prediction.cpu().numpy().tolist()
                    ground_truth=ground_truth.cpu().numpy().tolist()
                    common = Counter(prediction) & Counter(ground_truth)
                    num_same = sum(common.values())
                    if num_same == 0:
                        return 0
                    precision = 1.0 * num_same / len(prediction)
                    recall = 1.0 * num_same / len(ground_truth)
                    f1 = (2 * precision * recall) / (precision + recall)
                    return f1

                inputs = {
                    "input_ids": dev_batch[0],
                    "attention_mask": dev_batch[1],
                    "token_type_ids": dev_batch[2],
                }
                outputs = qa_model(**inputs)
                pred_start_indices = torch.argmax(outputs[0], dim=1)
                pred_end_indices = torch.argmax(outputs[1], dim=1)
                for i in range(len(dev_batch[0])):
                    pred_start=pred_start_indices[i]
                    pred_end = pred_end_indices[i]
                    gt_start=dev_batch[4][i]
                    gt_end = dev_batch[5][i]
                    prediction=dev_batch[0][i][gt_start:gt_end+1]
                    ground_truth=dev_batch[0][i][pred_start:pred_end+1]
                    total_reward+=f1_score(prediction,ground_truth)

    return 100.0 * total_reward / len(dev_dataset)


def create_optimizer_and_scheduler(model, args, num_training_steps, learning_rate):
    """
    Setup the optimizer and the learning rate scheduler.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if
                       any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=learning_rate,
        eps=args.adam_epsilon,
    )
    if num_training_steps:
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer,lr_scheduler
    else:
        return optimizer


def train_qa(args, model,train_dataloader, optimizer):
    total_loss=0
    for step_i, batch_i in enumerate(train_dataloader):
        model.train()

        inputs = {
            "input_ids": batch_i[0],
            "attention_mask": batch_i[1],
            "token_type_ids": batch_i[2],
            "start_positions": batch_i[3],
            "end_positions": batch_i[4],
            "input_values": batch_i[5],
        }

        outputs = model(**inputs)
        loss = outputs[0]

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        total_loss+=loss.item()

        if args.fp16:
            from apex import amp
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        if args.fp16:
            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        optimizer.step()
        model.zero_grad()

    return total_loss/len(train_dataloader)

def train(args, train_dataset, dev_dataset, qve_model, qa_model, tokenizer):

    # Model re-init
    # Seed must be set before instantiating the model when using model_init.
    set_seed(args)

    args.train_qve_batch_size = args.per_gpu_train_qve_batch_size * max(1, args.n_gpu)
    args.train_qa_batch_size = args.per_gpu_train_qa_batch_size * max(1, args.n_gpu)

    # Data loader and number of training steps
    if args.max_steps > 0:
        train_sampler = RandomSampler(train_dataset, replacement=True,
                                      num_samples=args.train_qve_batch_size * args.max_steps)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_qve_batch_size)

    num_update_steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)

    if args.max_steps > 0:
        t_total_qve = args.max_steps
        num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
            args.max_steps % num_update_steps_per_epoch > 0
        )
        
    else:
        t_total_qve = int(num_update_steps_per_epoch * args.num_train_epochs)
        num_train_epochs = args.num_train_epochs
        args.max_steps = t_total_qve


    optimizer_qve, lr_scheduler_qve = create_optimizer_and_scheduler(qve_model, args, t_total_qve, args.qve_learning_rate)
    optimizer_qa = create_optimizer_and_scheduler(qa_model, args, None, args.learning_rate)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")

        qve_model, optimizer_qve = amp.initialize(qve_model, optimizer_qve, opt_level=args.fp16_opt_level)
        qa_model, optimizer_qa = amp.initialize(qa_model, optimizer_qa, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        qve_model = torch.nn.DataParallel(qve_model)
        qa_model = torch.nn.DataParallel(qa_model)

    total_train_batch_size_qve = (
            args.train_qve_batch_size
            * args.gradient_accumulation_steps
    )
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", num_train_epochs)
    logger.info("  Instantaneous batch size per device QVE = %d, QA = %d", args.per_gpu_train_qve_batch_size, args.per_gpu_train_qa_batch_size,)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) QVE = %d, QA = %d", total_train_batch_size_qve, args.train_qa_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps QVE = %d", t_total_qve)

    global_step = 0
    epochs_trained = 0

    tr_loss_qa = torch.tensor(0.0).to(args.device)
    tr_loss_qve = torch.tensor(0.0).to(args.device)
    tr_reward = torch.tensor(0.0).to(args.device)
    tr_qa_acc = torch.tensor(0.0).to(args.device)

    best_reward = -100000
    lowest_loss = 100000

    logging_qa_loss_scalar,logging_qve_loss_scalar,logging_reward_scalar,logging_qa_acc_scalar  = 0.0, 0.0, 0.0, 0.0
    baseline_performance=cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)

    qa_model.zero_grad()
    qve_model.zero_grad()

    train_pbar = trange(epochs_trained, int(np.ceil(num_train_epochs)), desc="Epoch")

    qa_model_init_statedict = copy.deepcopy(qa_model).state_dict()

    for epoch in range(epochs_trained, int(np.ceil(num_train_epochs))):

        epoch_iterator = train_dataloader

        epoch_pbar = tqdm(epoch_iterator, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            qve_model.train()

            batch = tuple(t.to(args.device) for t in batch)

            threshold1 = (1-batch[9]).mean(dim=1).sort()[0][int(len(batch[9])*(1-args.selected_question_percentage))]
            noisy_GT = (1-batch[9]).mean(dim=1) >= threshold1

            inputs = {
                "input_ids": batch[6],
                "attention_mask": batch[7],
                "token_type_ids": batch[8],
                "marginal_info": batch[9],
            }

            qa_values = qve_model(**inputs)[0]

            if qa_values.size(1) == 1:
                qa_values = qa_values.view(-1)
            else:
                qa_values = torch.softmax(qa_values, dim=1)[:, 1].view(-1)

            #normalize qa values
            qa_values = (qa_values - qa_values.min())/(qa_values.max()-qa_values.min())

            # eval the QVE based on the noisy labels by the QA answerablility:
            # we deem top 60% (based on QA prob) inside the batch as positives and the left 40% as negatives
            # and calculate the accuracy as a signal to roughly watch QVE's training performance
            threshold2 = qa_values.sort()[0][int(len(batch[9])*(1-args.selected_question_percentage))]
            estimated_label = qa_values >= threshold2

            noise_acc_qve = 1.0 * (noisy_GT == estimated_label).sum() / len(noisy_GT)

            # sample the selection probability
            select_prob = Binomial(1, qa_values).sample()

            # train QA model
            inputs = TensorDataset(batch[0], batch[1], batch[2], batch[4], batch[5], select_prob)
            train_QA_sampler = RandomSampler(inputs)
            train_QA_dataloader = DataLoader(inputs, sampler=train_QA_sampler, batch_size=args.train_qa_batch_size)

            qa_loss = train_qa(args, qa_model, train_QA_dataloader, optimizer_qa)
            tr_loss_qa += qa_loss

            cur_qa_performance = cal_reward_func(args, dev_dataset, qa_model, type=args.reward_type)
            tr_qa_acc += cur_qa_performance
            reward = cur_qa_performance - baseline_performance

            tr_reward += reward

            epsilon = 1e-8  # avoid overflow
            threshold = 0.8  # Encourages exploration
            prob = select_prob * torch.log(qa_values + epsilon) + (1 - select_prob) * torch.log(1 - qa_values + epsilon)
            qve_loss_rl = -reward * prob.mean()
            qve_loss_aux = torch.relu(qa_values.mean() - threshold) + torch.relu(1 - threshold - qa_values.mean())

            qve_loss_total = qve_loss_rl + qve_loss_aux

            if args.gradient_accumulation_steps > 1:
                qve_loss_total = qve_loss_total / args.gradient_accumulation_steps
            tr_loss_qve+=qve_loss_total

            if args.fp16:
                with amp.scale_loss(qve_loss_total, optimizer_qve) as scaled_loss:
                    scaled_loss.backward()
            else:
                qve_loss_total.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    len(epoch_iterator) <= args.gradient_accumulation_steps
                    and (step + 1) == len(epoch_iterator)
            ):

                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer_qve), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(qve_model.parameters(), args.max_grad_norm)

                optimizer_qve.step()
                lr_scheduler_qve.step()
                qve_model.zero_grad()

                global_step += 1
                epoch = epoch + (step + 1) / len(epoch_iterator)

                if (args.logging_steps > 0 and global_step % args.logging_steps == 0):
                    logs = {}
                    tr_loss_qa_scalar = tr_loss_qa.item()
                    tr_loss_qve_scalar=tr_loss_qve.item()
                    tr_qa_acc_scalar=tr_qa_acc.item()
                    tr_reward_scalar=tr_reward.item()
                    logs["qa_loss"] = (tr_loss_qa_scalar - logging_qa_loss_scalar) / args.logging_steps
                    logs["eval_qa_current"] = (tr_qa_acc_scalar - logging_qa_acc_scalar) / args.logging_steps
                    logs["reward"] = (tr_reward_scalar - logging_reward_scalar) / args.logging_steps
                    logs["qve_loss_total"] = (tr_loss_qve_scalar - logging_qve_loss_scalar) / args.logging_steps
                    logs["noise_acc_qve"] = noise_acc_qve.item()
                    logs['eval_qa_baseline'] = baseline_performance
                    logs['num of selected questions'] = select_prob.sum().item()

                    logging_qa_loss_scalar = tr_loss_qa_scalar
                    logging_qve_loss_scalar = tr_loss_qve_scalar
                    logging_reward_scalar = tr_reward_scalar
                    logging_qa_acc_scalar = tr_qa_acc_scalar

                    logger.info(logs)

                    save_flag = False
                    if logs["reward"] > best_reward:
                        best_reward = logs["reward"]
                        output_dir = os.path.join(args.output_dir, "checkpoint-best-reward")
                        save_flag = True

                    if logs["qa_loss"] < lowest_loss:
                        lowest_loss = logs["qa_loss"]
                        output_dir = os.path.join(args.output_dir, "checkpoint-best-loss")
                        save_flag = True

                    if save_flag:
                        # Take care of distributed/parallel training
                        model_to_save = qve_model.module if hasattr(qve_model, "module") else qve_model
                        model_to_save.save_pretrained(output_dir)
                        tokenizer.save_pretrained(output_dir)

                        torch.save(args, os.path.join(output_dir, "training_args.bin"))

                        logger.info("Step: %d, Saving qve_model checkpoint to %s", global_step, output_dir)

                qa_model.load_state_dict(qa_model_init_statedict)


            if global_step % args.save_steps == 0:
                output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                # Take care of distributed/parallel training
                model_to_save = qve_model.module if hasattr(qve_model,"module") else qve_model
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving qve_model checkpoint to %s", output_dir)

                torch.save(optimizer_qve.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                torch.save(lr_scheduler_qve.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                logger.info("Saving optimizer and scheduler states to %s", output_dir)


            epoch_pbar.update(1)
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
        epoch_pbar.close()
        train_pbar.update(1)

        if args.max_steps > 0 and global_step >= args.max_steps:
            break

    train_pbar.close()

    logger.info("\n\nTraining completed.\n\n")

    return

def get_train_examples(filename):
    with open(filename) as f:
        lines = f.readlines()
    examples = []
    for line in lines:
        e = json.loads(line.strip())
        qas_id = e['id']
        question_text = e['question']
        context_text = e['context']
        answer_text = e['answers']['text'][0]
        is_impossible = e.get("is_impossible", False)
        start_position_character=e['answers']['answer_start'][0]
        answers = []
        for i, text in enumerate(e["answers"]["text"]):
            answers.append(
                {
                "text": text,
                "answer_start":  e["answers"]["answer_start"][i]
                })
        
        example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=" ",
                        is_impossible=is_impossible,
                        answers=answers)
        examples.append(example)
    return examples
                   
    
def load_and_cache_examples(args, tokenizer, qa_model_marginal=None, output_examples=False, dev=False):

    # Load data features from cache or dataset file
    mode="dev" if dev else "train"

    qa_dataset_name = [n for n in ["NewsQA","NaturalQuestionsShort","HotpotQA","TriviaQA-web"] if n in args.train_file][0]
    cached_features_file = os.path.join(
        "cached_{}_{}_{}".format(
            mode,
            qa_dataset_name,
            str(args.max_seq_length),
        ),
    )

    # Init features and dataset from cache if it exists
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset["features"],
            features_and_dataset["dataset"],
            features_and_dataset["examples"],
        )
    else:
        process_file = args.dev_file if dev else args.train_file
        logger.info("Creating features from dataset file at: %s", process_file)

        # processor = SquadV1Processor()
        # examples = processor.get_train_examples(data_dir=None, filename=process_file)
        
        examples = get_train_examples(process_file)

        features, dataset = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            return_dataset="pt",
            threads=args.threads,
        )

        if qa_model_marginal and not dev:
            marginal_info = calculate_marginal_info(qa_model_marginal, dataset, args)
            new_data = dataset.tensors + (marginal_info.cpu(),)
            dataset = TensorDataset(*new_data)

        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save({"features": features, "dataset": dataset, "examples": examples}, cached_features_file)

    if output_examples:
        return dataset, examples, features
    return dataset

def write_filtered_qa_examples(filtered_id_list, train_file, output_dir):
    
    examples = get_train_examples(train_file)
    print(examples[0])
    filtered_examples = []
    with open(train_file,'r',encoding='utf-8') as f:
        for line in f:
            example = json.loads(line.strip())
            if example['id'] in filtered_id_list:
                filtered_examples.append(line.strip())
                
    out_file = os.path.join(output_dir, "filtered_qa.jsonl")
    with open(out_file, 'w') as f:
        f.writelines([f"{l}\n" for l in filtered_examples])
    
    print("Wrote", out_file)
    
    

def estimation(args, tokenizer, qve_model, qa_model):
    dataset, examples, features = load_and_cache_examples(args, tokenizer, qa_model, output_examples=True, dev=False)
    if args.n_gpu > 0:
        qve_model=torch.nn.DataParallel(qve_model)
    qve_model.eval()
    args.dev_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    data_sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=data_sampler, batch_size=args.dev_batch_size)

    # Eval!
    logger.info("***** Running Question Value Estimation *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.dev_batch_size)

    all_qa_values = []
    all_qa_answerabilities = []
    for batch in tqdm(dataloader, desc="Estimation"):
        batch = tuple(t.to(args.device) for t in batch)
        with torch.no_grad():
            inputs = {
                "input_ids": batch[6],
                "attention_mask": batch[7],
                "token_type_ids": batch[8],
                "marginal_info": batch[9],
            }

            qa_values = qve_model(**inputs)[0]

            if qa_values.size(1) == 1:
                qa_values = qa_values.view(-1)
            else:
                qa_values = torch.softmax(qa_values, dim=1)[:, 1].view(-1)

            all_qa_values.append(qa_values)
            all_qa_answerabilities.append((1 - batch[9]).mean(dim=1))

    all_qa_values = torch.cat(all_qa_values)
    all_qa_answerabilities = torch.cat(all_qa_answerabilities)

    # all_qa_values = Binomial(1, all_qa_values).sample()
    all_qa_values = all_qa_values.cpu().detach().numpy()

    qid2feature = {}
    for ii, feature in enumerate(features):
        if feature.qas_id not in qid2feature:
            qid2feature[feature.qas_id] = [ii]
        else:
            qid2feature[feature.qas_id].append(ii)

    qid2qv = {}
    for qid, fids in qid2feature.items():
        qid2qv[qid] = max(all_qa_values[fids])

    filtered_id_list = list(dict(sorted(qid2qv.items(),key = lambda x: x[1], reverse=True)).keys())[:int(len(qid2qv)*args.selected_question_percentage)]

    write_filtered_qa_examples(filtered_id_list, args.train_file, args.output_dir)

    # # filtered_id_list = [qid for qid,qv in qid2qv.items() if qv==1]
    # ##write to json
    # data_json = json.load(open(args.train_file, 'r'))
    # new_passages_train = []

    # for passages in data_json['data']:
    #     new_paras_train = []

    #     for para in passages['paragraphs']:
    #         context = para['context']
    #         new_qas_train = []

    #         for qa in para['qas']:
    #             if qa['id'] in filtered_id_list:
    #                 new_qas_train.append(qa)

    #         if len(new_qas_train) > 0:
    #             new_paras_train.append({'context': context, 'qas': new_qas_train})

    #     if len(new_paras_train) > 0:
    #         new_passages_train.append({'title': passages['title'], 'paragraphs': new_paras_train})

    # filtered_data_json = {'data': new_passages_train, 'version': data_json['version']}

    # total = 0
    # context_num = 0
    # for paras in data_json['data']:
    #     for para in paras['paragraphs']:
    #         context_num += 1
    #         qa_num = len(para['qas'])
    #         total += qa_num
    # logger.info('Before filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

    # total = 0
    # context_num = 0
    # for paras in filtered_data_json['data']:
    #     for para in paras['paragraphs']:
    #         context_num += 1
    #         qa_num = len(para['qas'])
    #         total += qa_num
    # logger.info('After filtering: Train QA Num: %d, Total Context: %d' % (total, context_num))

    # json.dump(filtered_data_json, open(os.path.join(args.output_dir, "filtered_qa.json"), 'w'))

    return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--qa_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--qve_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained question value estimator",
    )
    parser.add_argument(
        "--marginal_model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints and predictions will be written.",
    )

    parser.add_argument(
        "--train_file",
        default=None,
        type=str,
        help="The input training file.",
    )
    parser.add_argument(
        "--dev_file",
        default=None,
        type=str,
        help="The input dev file (target annotations) to provide feedback for QVE training. ",
    )

    parser.add_argument(
        "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O2",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    parser.add_argument(
        "--max_seq_length",
        default=384,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
             "longer than this will be truncated, and sequences shorter than this will be padded.",
    )
    parser.add_argument(
        "--doc_stride",
        default=128,
        type=int,
        help="When splitting up a long document into chunks, how much stride to take between chunks.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum number of tokens for the question. Questions longer than this will "
             "be truncated to this length.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_estimation", action="store_true", help="Whether to question value estimation for the training set")

    parser.add_argument(
        "--do_lower_case", action="store_true", help="Set this flag if you are using an uncased model."
    )

    parser.add_argument(
        "--gradient_checkpointing", action="store_true", help="Set this flag if you are using gradient checkpointing"
    )

    parser.add_argument("--per_gpu_train_qve_batch_size", default=60, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_train_qa_batch_size", default=6, type=int, help="Batch size per GPU/CPU for QA training.")

    parser.add_argument("--reward_type", default="exact", type=str, help="reward type: exact/f1/loss")
    parser.add_argument("--sliding_window_size", default=5, type=int,
                        help="sliding window size for updating baseline of reinforced algo.")

    parser.add_argument(
        "--per_gpu_eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for evaluation."
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--qve_learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--qve_eval_data_num", default=-1, type=int,
                        help="how many target data used for calculating reinforced loss for qve. -1 means all")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=3.0, type=float, help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")

    parser.add_argument(
        "--selected_question_percentage", default=0.6, type=float, help="how many questions to select?"
    )

    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")

    parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
    parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--threads", type=int, default=24, help="multiple threads for converting example to features")

    parser.add_argument("--add_marginal_info", action="store_true", help="Whether not to add marginal info to qve model")


    args = parser.parse_args()


    if args.doc_stride >= args.max_seq_length - args.max_query_length:
        logger.warning(
            "WARNING - You've set a doc stride which may be superior to the document length in some "
            "examples. This could result in errors when building features from the examples. Please reduce the doc "
            "stride or increase the maximum length to ensure the features are correctly built."
        )

    if (
            os.path.exists(args.output_dir)
            and os.listdir(args.output_dir)
            and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.warning(
        "device: %s, n_gpu: %s, fp16: %s",
        device,
        args.n_gpu,
        args.fp16,
    )

    # Set seed
    set_seed(args)

    if args.do_train:

        args.reward_type = args.reward_type.lower()

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.qve_model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
            use_fast=False,
        )

        config = AutoConfig.from_pretrained(
            args.marginal_model_name_or_path,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        qa_model_marginal = ModelForDownstreamTasks.from_config(
            config,
            args.marginal_model_name_or_path,
            task_heads=EXTRACTIVE_HEAD,
        )
        qa_model_marginal.set_task_head(next(iter(EXTRACTIVE_HEAD)))

        config = AutoConfig.from_pretrained(
            args.qa_model_name_or_path,
            gradient_checkpointing=args.gradient_checkpointing,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        qa_model = ModelForDownstreamTasks.from_config(
            config,
            args.qa_model_name_or_path,
            task_heads=EXTRACTIVE_HEAD,
        )
        qa_model.set_task_head(next(iter(EXTRACTIVE_HEAD)))


        config = AutoConfig.from_pretrained(
            args.qve_model_name_or_path,
            num_labels=2,
            gradient_checkpointing=args.gradient_checkpointing,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        config.marginal = args.add_marginal_info  # add marginal information for QVE

        qve_model = BertForSequenceClassification.from_pretrained(
            args.qve_model_name_or_path,
            from_tf=bool(".ckpt" in args.qve_model_name_or_path),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        if '<ANS>' not in tokenizer.additional_special_tokens:
            special_tokens_dict = {'additional_special_tokens': ['<ANS>', '<NULL_ANS>']}
            tokenizer.add_special_tokens(special_tokens_dict)
            logger.info("Adding Special Tokens: %s", special_tokens_dict)
            qve_model.resize_token_embeddings(len(tokenizer))

        qa_model.to(args.device)
        qve_model.to(args.device)
        qa_model_marginal.to(args.device)

        logger.info("Training/evaluation parameters %s", args)
        train_dataset = load_and_cache_examples(args, tokenizer, qa_model_marginal,output_examples=False, dev=False)
        dev_dataset= load_and_cache_examples(args, tokenizer, output_examples=False, dev=True)
        train(args, train_dataset, dev_dataset, qve_model, qa_model, tokenizer)

        # Save the trained model and the tokenizer
        logger.info("Saving model checkpoint to %s", args.output_dir)

        model_to_save = qve_model.module if hasattr(qve_model,"module") else qve_model
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.do_estimation:
        if args.do_train:
            logger.info("Loading checkpoints saved during training for evaluation")
            checkpoints = args.output_dir
        else:
            logger.info("Loading checkpoint %s for evaluation", args.qve_model_name_or_path)
            checkpoints = args.qve_model_name_or_path

        config = AutoConfig.from_pretrained(
            checkpoints,
            gradient_checkpointing=False,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.qve_model_name_or_path,
            do_lower_case=args.do_lower_case,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        qve_model = BertForSequenceClassification.from_pretrained(checkpoints,config=config)

        config = AutoConfig.from_pretrained(
            args.qa_model_name_or_path,
            gradient_checkpointing=args.gradient_checkpointing,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )

        qa_model = ModelForDownstreamTasks.from_config(
            config,
            args.qa_model_name_or_path,
            task_heads=EXTRACTIVE_HEAD,
        )
        qa_model.set_task_head(next(iter(EXTRACTIVE_HEAD)))

        qa_model.to(args.device)
        qve_model.to(args.device)

        estimation(args, tokenizer, qve_model, qa_model)

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained(
    #         "/dccstor/bsiyer6/public/qve/primeqa/examples/QVE/checkpoints/TriviaQA-web_QVE_base/checkpoint-500/",
    #         do_lower_case=False,
    #         cache_dir=None,
    #         use_fast=False,
    #     )
    # examples = get_train_examples("/dccstor/bsiyer6/public/qve/primeqa/examples/QVE/data/Trivia-web_QG/TriviaQA-web.train.targetfinetuned.gen.jsonl")
    # features, dataset = squad_convert_examples_to_features(
    #         examples=examples,
    #         tokenizer=tokenizer,
    #         max_seq_length=384,
    #         doc_stride=128,
    #         max_query_length=64,
    #         is_training=True,
    #         return_dataset="pt",
    #         threads=24,
    #     )

    main()
