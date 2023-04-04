import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
from transformers import GPT2Config
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import json
import torch.optim as optim
from tqdm import trange, tqdm
import math
from datetime import datetime
from primeqa.mitqa.utils.hybridqa_utils import whitelist, is_year
import sys
import copy
import pdb
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder, util



def set_seed(args):
    np.random.seed(args.seed_lg)
    torch.manual_seed(args.seed_lg)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed_lg)
        
def get_top_k_passages_from_corpus(passage_embeddings,passage_id_dict,doc_retriever,query,top_k):
    """
    The get_top_k_passages_from_corpus function takes in a list of passage embeddings, the dictionary mapping
    passage IDs to their corresponding passages, and the query. It returns a list of relevant sentences from 
    the corpus that are most similar to the query.
    
    Args:
        passage_embeddings: Store the embeddings of all passages in the corpus
        passage_id_dict: Map the passage_id to the actual text
        doc_retriever: Retrieve the top k passages from the corpus
        query: Search for the query in the corpus
        top_k: Specify the number of passages to return
    
    Returns:
        The top k passages from the corpus that are most relevant to the query
    """
    
    corpus_embeddings = util.normalize_embeddings(passage_embeddings)

    query_embeddings = doc_retriever.encode([query], convert_to_tensor=True)
    query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, top_k=top_k, score_function=util.dot_score)
    hits = hits[0]
    relevant_sents =[]
    for hit in hits:
        relevant_sents.append(passage_id_dict[hit['corpus_id']])
    return relevant_sents

def get_links(mapping,row_id):
    """
    The get_links function takes in a dictionary mapping and a row_id. It then checks if the row_id is present in the
    dictionary as key. If it is, it returns all of the links associated with that key (which are stored as values). If not,
    it returns an empty list.
    
    Args:
        mapping: Map the row_id to the links
        row_id: Get the row_id from the dataframe
    
    Returns:
        The list of links for a given row_id
    """
    links =[]
    v = mapping[row_id] if row_id in mapping else []
    links = [i.replace(" ","_")for i in v]
    links = ["/wiki/"+i for i in links]
    return links

def sample_sequence(model, length, context, args, num_samples=1, temperature=1, stop_token=None, \
                    top_k=0, top_p=0.0, device='cuda'):
    """
    The sample_sequence function takes in a model, length of output sequence, context (input), args (additional parameters)
    and returns the generated sequence. The context is concatenated with the token_id for each token in the output sequence.
    The function also handles batching and padding.
    
    Args:
        model: Pass the model to sample_sequence
        length: Specify the length of the text that we want to generate
        context: Provide the model with a starting sentence
        args: Pass in the following additional arguments:
        num_samples: Determine how many samples to generate
        temperature: Control the randomness of the generated text
        stop_token: Determine when the text generation is stopped
        \
                        top_k: Control the number of highest probability vocabulary tokens to consider at each step
        top_p: Control the &quot;randomness&quot; of the sample
        device: Tell the model whether you are using a cpu or gpu
    
    Returns:
        The generated sequence
    """
    if isinstance(context, list):
        context = torch.tensor(context, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)

    generated = context
    batch_size = generated.shape[0]
    
    finished_sentence = [False for _ in range(batch_size)]
    with torch.no_grad():
        for _ in range(length):
            outputs = model(generated, *args)
            if isinstance(outputs, list) or isinstance(outputs, tuple):
                print(isinstance(outputs, list) or isinstance(outputs, tuple))
                next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            else:
                next_token_logits = outputs.logits[:, -1, :] / (temperature if temperature > 0 else 1.)

            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)

            if all(finished_sentence):
                break

    return generated

def load_all_tables():
    data = json.load(open("data/ottqa/all_plain_tables.json"))
    return data 

def train_link_generator(args):
    """
    The train_link_generator function trains a model to generate links between two entities.
    The function takes in the following parameters:
        - args (argparse): A namespace containing the arguments used to train the model. 
          These arguments are passed from train_link_generator's caller, most likely run_experiment.py. 
    
    Args:
        args: Pass in the parameters like batch_size, learning rate etc
    
    """
    if torch.cuda.is_available():
        args.device_lg = torch.device("cuda")
        args.n_gpu = torch.cuda.device_count()
    else:
        args.device_lg= torch.device("cpu")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.add_tokens(['[SEP]', '[EOS]', '[START]', '[ENT]'])
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    print("Start Training.")
    model = nn.DataParallel(model)
    model.to(args.device_lg)
    recording_time = datetime.now().strftime('%m_%d_%H_%M')
    tb_writer = SummaryWriter(log_dir='link_generator/{}'.format(recording_time))
    dataset = LinkGenearationDataset(args.dataset, 'train', tokenizer, args.max_source_len, args.max_target_len)
    optimizer = optim.Adam(model.parameters(), args.learning_rate_lg)
    
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.batch_size_lg, num_workers=8, pin_memory=True, drop_last=True)
    print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(train_dataloader)))
    
    avg_loss = 0
    global_step = 0
    for epoch in trange(args.num_epoch_lg):
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, indexed_batch in enumerate(epoch_iterator):
            model.train()

            batch = tuple(t.to(args.device_lg) for t in indexed_batch[2:])

            prefix, trg_inp, trg_out, mask = batch

            inputs = torch.cat([prefix, trg_inp], 1)

            model.zero_grad()
            optimizer.zero_grad()

            logits = model(inputs)[0]
            logits = logits[:, -trg_out.shape[1]:, :].contiguous()

            loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))

            loss = loss * mask.view(-1)
            loss = loss.sum() / mask.sum()

            avg_loss += loss.item()

            loss.backward()
            optimizer.step()
            global_step += 1

            if step % args.every == 0 and step > 0:
                model.eval()
                tb_writer.add_scalar("loss", math.exp(avg_loss / args.every), global_step)
                prefix = torch.cat([prefix, trg_inp[:, :1]], -1)
                prefix = prefix[:2]

                gt_inputs = trg_out.cpu().data.numpy()[:2]

                samples = sample_sequence(model, 16, prefix, [], 1, temperature=0)
                samples = samples[:, prefix.shape[1]:]
                samples = samples.cpu().data.numpy()
                prefix = prefix.cpu().data.numpy()

                for p, s, gt in zip(prefix, samples, gt_inputs):
                    text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    text = text[: text.find('[EOS]')]
                    pre_text = tokenizer.decode(p, clean_up_tokenization_spaces=True)
                    print("Input |||||| ", pre_text)
                    print("PREDICTION |||||| ", text)
                    text = tokenizer.decode(gt, clean_up_tokenization_spaces=True)
                    text = text[: text.find('[EOS]')]
                    print("GROUNDTRUH |||||| ",text)
                    break

                avg_loss = 0

        torch.save(model.module.state_dict(), 'link_generator/model-ep{}.pt'.format(epoch))
    return avg_loss

def predict_link_for_tables(args,retrieved_data,doc_retriever):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.add_tokens(['[SEP]', '[EOS]', '[START]', '[ENT]'])
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))
    table_dict = {}
    all_tables = load_all_tables()
    for d in retrieved_data:
        table_dict[d['table_id']]= all_tables[d['table_id']]    
        
    dataset = LinkGenearationDataset(table_dict, 'custom', tokenizer, args.max_source_len, args.max_target_len, args.shard)
    sampler = SequentialSampler(dataset)
    
    dev_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_lg, num_workers=8, pin_memory=True, drop_last=True)        
    print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(dev_dataloader)))
    
    model.load_state_dict(torch.load(args.linker_model))
    model = nn.DataParallel(model)
    model.to(args.device_lg)
    model.eval()
    print("Loaded model from {}".format(args.linker_model))

    mapping = {}
    for indexed_batch in tqdm(dev_dataloader, desc="Decoding"):
        batch = tuple(t.to(args.device_lg) for t in indexed_batch[2:])
        row_ids = indexed_batch[0]
        links = indexed_batch[1]

        prefix, trg_inp, trg_out, mask = batch
        prefix = torch.cat([prefix, trg_inp[:, :1]], -1)
        samples = sample_sequence(model, 16, prefix, [], 1, temperature=0)
        samples = samples[:, prefix.shape[1]:]
        samples = samples.cpu().data.numpy()
        for row_id, link, s in zip(row_ids, links, samples):
            text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
            decoded = []
            for _ in text[:text.find('[EOS]')].split(' # '):
                name = _.replace('#', '').strip()
                if len(name) > 1 and name not in decoded:
                    decoded.append(name)
            mapping[row_id] = mapping.get(row_id, []) + decoded
            continue

    for k, v in dataset.mapping.items():
        mapping[k] = v if k not in mapping else  mapping[k].extend(v)

    
    new_data = []
    for d in retrieved_data:
        table_id = d['table_id']
        table_data = all_tables[d['table_id']]
        table_links =[]
        for i,r in enumerate(table_data['data']):
            row_id = table_id+"_"+str(i)
            row_links = get_links(mapping,row_id)
            table_links.append(row_links)

        d['row_passage_links'] = table_links
        d["table"] = table_data
        new_data.append(d)
    return new_data
    


class LinkGenearationDataset(Dataset):
    def __init__(self, datapath, option, tokenizer, source_max_len, target_max_len, shards=None):
        super(LinkGenearationDataset, self).__init__()
        self.tokenizer = tokenizer
        self.source_max_len = source_max_len
        self.target_max_len = target_max_len
        self.option = option
        self.mapping = {}
        assert option in ['train', 'dev', 'all','custom']
        if option != 'all' and option != 'custom':
            with open('data/ottqa/released_data/train_dev_test_table_ids.json', 'r') as f:
                table_ids = set(json.load(f)[option])

        if isinstance(datapath,dict):
            tables = datapath
        else:
            with open(datapath) as f:
                tables = json.load(f)

        if self.option == 'all':
            assert shards is not None
            index, total_shard = [int(_) for _ in shards.split('@')]
            table_ids = list(tables.keys())
            length = len(table_ids) // total_shard
            table_ids = table_ids[index * length : (index+1) * length]
            print("Running {} out of shard {}".format(index, total_shard))
            table_ids = set(table_ids)

        self.data = []
        for k, table in tables.items():
            title = table['title']
            sec_title = table['section_title']

            if isinstance(table['header'][0], list):
                headers = [_[0] for _ in table['header']]
            else:
                headers = table['header']

            for i, row in enumerate(table['data']):
                row_id = k + '_{}'.format(i)
                for header, cell in zip(headers, row):
                    content = cell[0] if isinstance(cell, list) else cell
                    assert isinstance(content, str)
                    if not whitelist(content):
                        continue

                    inputs = 'In ' + title + ' [SEP] ' + sec_title + ' [SEP] ' + header + ' [ENT] ' + content + ' [ENT] '
                    links = []
                    if isinstance(cell, list):
                        for link in cell[1]:
                            links.append(link.replace('/wiki/', '').replace('_', ' '))
                    else:
                        # For plain tables
                        pass
                    outputs = ' # '.join(links)
                    self.data.append((row_id, inputs, outputs))

    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row_id, inputs, outputs = self.data[index]
        links = copy.deepcopy(outputs)

        prefix = self.tokenizer.encode(inputs, add_special_tokens=False)
        prefix = prefix[max(0, len(prefix) - self.source_max_len):]
        prefix = [self.tokenizer.eos_token_id] * (self.source_max_len - len(prefix)) + prefix

        outputs = self.tokenizer.encode('[START] ' + outputs + ' [EOS]', add_special_tokens=False)
        outputs = outputs[:self.target_max_len]
        outputs = outputs + [self.tokenizer.eos_token_id] * (self.target_max_len - len(outputs))
        trg_input = outputs[:-1]
        trg_output = outputs[1:]

        prefix = torch.LongTensor(prefix)
        trg_input = torch.LongTensor(trg_input)
        trg_output = torch.LongTensor(trg_output)            

        mask = (trg_output != self.tokenizer.eos_token_id).float()

        return row_id, links, prefix, trg_input, trg_output, mask