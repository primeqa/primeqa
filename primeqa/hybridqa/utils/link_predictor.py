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
from utils.hybridqa_utils import whitelist, is_year
import sys
import copy
import pdb




def set_seed(args):
    np.random.seed(args.seed_lg)
    torch.manual_seed(args.seed_lg)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed_lg)


def get_links(mapping,row_id):
    links =[]
    for k,v in mapping.items():
        if k.lower() == row_id.lower():
            links = [i.replace(" ","_")for i in v]
            links = ["/wiki/"+i for i in links]
    return links

def sample_sequence(model, length, context, args, num_samples=1, temperature=1, stop_token=None, \
                    top_k=0, top_p=0.0, device='cuda'):
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
    

def predict_link_for_tables(args,retrieved_data):
 
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
    
    model.load_state_dict(torch.load(args.load_from))
    model = nn.DataParallel(model)
    model.to(args.device_lg)
    model.eval()
    print("Loaded model from {}".format(args.load_from))

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
        if k not in mapping:
            mapping[k] = v
        else:
            mapping[k].extend(v)
    
    #index, shard = [int(_) for _ in args.shard.split('@')]
    new_data = []
    for d in retrieved_data:
        table_id = d['table_id']
        table_data = all_tables[d['table_id']]
        table_links =[]
        for i,r in enumerate(table_data['data']):
            row_id = table_id+"_"+str(i)
            # print(row_id)
            row_links = get_links(mapping,row_id)
            table_links.append(row_links)
        d['row_passage_links'] = table_links
        d["table"] = table_data
        new_data.append(d)
    
    return new_data
    #     f.write(json_str + '\n')
    # f.close()
    


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
            # if k not in table_ids:
            #     continue

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt2', type=str)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42, help="random seed for initialization")
    parser.add_argument('--dataset', default=None, type=str, help="Whether to use dataset")
    parser.add_argument('--load_from', default=None, type=str, help="Whether to use dataset")
    parser.add_argument('--batch_size', default=128, type=int, help="Whether to use dataset")
    parser.add_argument('--every', default=50, type=int, help="Whether to use dataset")
    parser.add_argument('--max_source_len', default=32, type=int, help="Whether to use dataset")
    parser.add_argument('--max_target_len', default=16, type=int, help="Whether to use dataset")
    parser.add_argument('--do_train_lg', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_all_lg', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--do_val_lg', default=False, action="store_true", help="whether to train or test the model")
    parser.add_argument('--learning_rate_lg', default=5e-6, type=float, help="whether to train or test the model")
    parser.add_argument('--shard', default=None, type=str, help="whether to train or test the model")

    args = parser.parse_args()

    args.device_lg = torch.device("cuda")
    args.n_gpu = torch.cuda.device_count()

    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.add_tokens(['[SEP]', '[EOS]', '[START]', '[ENT]'])
    model = GPT2LMHeadModel.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    if args.do_train_lg:
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
        for epoch in trange(10):
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

    if args.do_val_lg:
        dataset = LinkGenearationDataset(args.dataset, 'all', tokenizer, args.max_source_len, args.max_target_len)
        sampler = SequentialSampler(dataset)
        dev_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_lg, num_workers=0, pin_memory=True, drop_last=True)        
        print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(dev_dataloader)))
        
        model.load_state_dict(torch.load(args.load_from))
        model = nn.DataParallel(model)
        model.to(args.device_lg)
        model.eval()
        print("Loaded model from {}".format(args.load_from))

        succ, prec_total, recall_total = 0, 0, 0
        mapping = {}
        for step, indexed_batch in enumerate(dev_dataloader):
            batch = tuple(t.to(args.device_lg) for t in indexed_batch[2:])
            row_ids = indexed_batch[0]
            links = indexed_batch[1]

            prefix, trg_inp, trg_out, mask = batch
            prefix = torch.cat([prefix, trg_inp[:, :1]], -1)
            
            with torch.no_grad():
                samples = sample_sequence(model, 16, prefix, [], 1, temperature=0)

            samples = samples[:, prefix.shape[1]:]
            samples = samples.cpu().data.numpy()
            for row_id, link, s in zip(row_ids, links, samples):
                text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                
                decoded = []
                for _ in text[:text.find('[EOS]')].split(' # '):
                    name = _.replace('#', '').strip()
                    if len(name) > 1:
                        decoded.append(name)

                link = link.split(' # ')
                succ += len(set(link) & set(decoded))
                prec_total += len(decoded)
                recall_total += len(link)
                mapping[row_id] = mapping.get(row_id, []) + decoded

            precision = succ / prec_total
            recall = succ / recall_total
            f1 = 2 * precision * recall / (precision + recall)
            sys.stdout.write('finished {}/{} ratio {} \r'.format(step, len(dev_dataloader), f1))

        with open('link_generator/row_passage_query.json', 'w') as f:
            json.dump(mapping, f, indent=2)

    if args.do_all_lg:
        assert '@' in args.shard
        dataset = LinkGenearationDataset(args.dataset, 'all', tokenizer, args.max_source_len, args.max_target_len, args.shard)
        sampler = SequentialSampler(dataset)
        dev_dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size_lg, num_workers=8, pin_memory=True, drop_last=True)        
        print("Dataset Size = {}. Loader Size = {}".format(len(dataset), len(dev_dataloader)))
        
        model.load_state_dict(torch.load(args.load_from))
        model = nn.DataParallel(model)
        model.to(args.device_lg)
        model.eval()
        print("Loaded model from {}".format(args.load_from))

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
            if k not in mapping:
                mapping[k] = v
            else:
                mapping[k].extend(v)

        index, shard = [int(_) for _ in args.shard.split('@')]
        f = open('link_generator/row_passage_query.json-0000{}-0000{}'.format(index, shard), 'w')
        for k, v in mapping.items():
            json_str = json.dumps((k, v))
            f.write(json_str + '\n')
        f.close()