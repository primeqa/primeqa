from torch import cuda
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time
import json
import argparse
from tableQG.utils import load_wtq_tables, convert_sql_to_string

import logging
logging.basicConfig(level=logging.ERROR)

device = 'cuda' if cuda.is_available() else 'cpu'

# make it hyperparameter
# command line
MAX_LEN = 100
SUMMARY_LEN = 50
TRAIN_BATCH_SIZE = 10
VALID_BATCH_SIZE = 10
EPOCHS = 15
LEARNING_RATE = 1e-3

# model_name as hyperparameter
tokenizer = T5Tokenizer.from_pretrained("t5-base")
# config is necessary when loading finetuned T5-model
config = T5ForConditionalGeneration.from_pretrained('t5-base').config


class CustomDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.sql = self.data.sql
        self.question = self.data.question

    def __len__(self):
        return len(self.sql)

    def __getitem__(self, index):
        sql = str(self.sql[index])
        sql = ' '.join(sql.split())

        question = str(self.question[index])
        question = ' '.join(question.split())

        source = self.tokenizer.batch_encode_plus(
            [sql], max_length=self.source_len, pad_to_max_length=True, return_tensors='pt', truncation=True)
        target = self.tokenizer.batch_encode_plus(
            [question], max_length=self.summ_len, pad_to_max_length=True, return_tensors='pt', truncation=True)

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_ids.to(dtype=torch.long)
        }


def get_data(df, train_size=0.8):
    # train_size = 0.8
    train_dataset = df.sample(
        frac=train_size, random_state=42).reset_index(drop=True)
    test_dataset = df.drop(train_dataset.index).reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    training_set = CustomDataset(
        train_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN, SUMMARY_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                   'shuffle': False,
                   'num_workers': 0
                   }

    training_loader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)
    return training_loader, testing_loader


def train(model, training_loader, epoch, optimizer):
    start_time = time.time()

    if args.column_header:  # if to use column headers with the SQL as input
        if_col_header = 'col-header_'
    else:
        if_col_header = ''

    model.train()
    if not args.exp and args.output_dir:
        model_path=args.output_dir

    else:
        if args.group != 'all':
            model_path = './models/t5_'+if_col_header +\
                'nw-'+str(args.max_num_where)+'_if-agg-' + \
                str(args.if_agg)+'_group-'+str(args.group)
        else:
            model_path = './models/t5_'+if_col_header +\
                'nw-'+str(args.max_num_where)+'_if-agg-'+str(args.if_agg)

    for _, data in enumerate(training_loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask,
                        decoder_input_ids=y_ids, lm_labels=lm_labels)
        loss = outputs[0]

        if _ % 1000 == 0:
            t = time.time() - start_time
            print(
                f'Epoch: {epoch}, step: {_}, Loss:  {loss.item()}, time-taken: {t}')
            model.save_pretrained(model_path)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def writer(predictions, actuals, test_dataset):
    output_dict = []
    for i, p in enumerate(predictions):
        pdict = {'predictions': p}
        pdict['quesion'] = actuals[i]
        pdict['sql'] = test_dataset.iloc[i].sql
        output_dict.append(pdict)
    return output_dict


def validate(model, testing_loader):
    model.eval()
    predictions = []
    actuals = []
    num_sequences = 5
    start_time = time.time()
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=60,
                num_beams=5,
                repetition_penalty=3.5,
                length_penalty=1.0,
                early_stopping=True,
                num_return_sequences=num_sequences
            )

            preds = [tokenizer.decode(
                g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]
            if _ % 10 == 0:
                print(f'Completed {_}', time.time() - start_time)

            pred_list = []
            for i in range(0, len(preds), num_sequences):
                pred_list.append(preds[i:i+num_sequences])
            predictions.extend(pred_list)
            actuals.extend(target)
            # break
    return predictions, actuals


def inference(sql_dict, model, tokenizer, table=[], use_col=False):
    ''' Format of sql_dict = {'col':['select','Captain'],
    			'conds':[['Country','equal','India'], ['Sports','equal','Cricket']],
     			'answer': 'Kohli'} '''
    input_str = convert_sql_to_string(sql_dict, table, use_col)

    data = tokenizer.batch_encode_plus(
        [input_str], max_length=MAX_LEN, pad_to_max_length=True, return_tensors='pt', truncation=True)
    ids = data['input_ids'].to(device, dtype=torch.long)
    mask = data['attention_mask'].to(device, dtype=torch.long)

    generated_ids = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_length=60,
        num_beams=10,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
        num_return_sequences=5
    )
    preds = [tokenizer.decode(g, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True) for g in generated_ids]
    return preds


'''
 This function runs inference on the provided data file and generates question using pre-trained t5 module
'''
def run_test(args):
    data_file = args.test_data_path
    print('Loading data from', data_file)
    df = pd.read_csv(data_file, encoding='latin-1', sep='\t')

    df = df[['sql', 'question']]
    df.sql = 'generate question: ' + df.sql
    df.head()

    model = T5ForConditionalGeneration.from_pretrained(args.model_path)
    model = model.to(device)
    testing_loader, _ = get_data(df, 1.0)

    predictions, actuals = validate(model, testing_loader)
    test_dataset = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    output_dict = writer(predictions, actuals, test_dataset)

    with open(args.prediction_file_path, 'w') as fp:
        json.dump(output_dict, fp)
    print('Output Files generated for review')

''' 
    This function takes arguments as input and trains the t5 generation module. Users can provide
    parameters such as group -> Which is domain/topic, model path and data path etc. 
'''

def run_train(args):
    if args.column_header:  # if to use column headers with the SQL as input
        if_col_header = 'col-header_'
    else:
        if_col_header = ''

    if args.group != 'all':
        data_file = './data/train_qgen_data_' +\
            if_col_header + 'nw-' + str(args.max_num_where)+'_if-agg-'+str(args.if_agg) +\
            '_group-'+str(args.group)+'.csv'
    else:
        data_file = './data/train_qgen_data_' +\
            if_col_header + 'nw-' + \
            str(args.max_num_where)+'_if-agg-'+str(args.if_agg)+'.csv'
    print('Loading data from', data_file)
    df = pd.read_csv(data_file, encoding='latin-1', sep='\t')

    df = df[['sql', 'question']]
    df.sql = 'generate question: ' + df.sql
    # df.text += '</s>'
    # for i in range(len(df)):
    # 	df.ctext[i] = df.ctext[i].replace('.', ' </s>')
    df.head()

    train_size = 0.95

    if args.t5_path == '':
        model = T5ForConditionalGeneration.from_pretrained('t5-base')
        t5_name = ''
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            args.t5_path, config=config)
        t5_name = args.t5_path.replace('pytorch_model_', '')
        t5_name = t5_name.replace('.bin', '')
        t5_name = 'FT-' + '-'.join(t5_name.split('/')[-3:]) + '_'

    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)
    training_loader, testing_loader = get_data(df, train_size)

    start_time = time.time()
    #Todo: 
    if args.group != 'all':
        model_path = './models/t5_'+t5_name+if_col_header +\
            'nw-' + str(args.max_num_where)+'_if-agg-' + \
            str(args.if_agg)+'_group-'+str(args.group)
    else:
        model_path = './models/t5_' + t5_name +\
            if_col_header+'nw-' + \
            str(args.max_num_where) + '_if-agg-' + str(args.if_agg)
    print('Model will be saved in ', model_path)
    for epoch in range(EPOCHS):
        train(model, training_loader, epoch, optimizer)

    model.save_pretrained(model_path)

    print('Trained model saved')


def generate_questions_group(args):
    if args.column_header:  # if to use column headers with the SQL as input
        if_col_header = 'col-header_'
    else:
        if_col_header = ''

    if args.split == 'train':
        split_prefix = ''
    else:
        split_prefix = '_dev'

    # loading the model
    if args.model_path != '':
        model_path = './models/' + args.model_path
    else:
        if args.group != 'all':
            model_path = './models/t5_'+if_col_header +\
                'nw-'+str(args.max_num_where)+'_if-agg-' + \
                str(args.if_agg)+'_group-'+str(args.group)
        else:
            # model_path = './models/t5_'+if_col_header+\
            # 	'nw-'+str(args.max_num_where)+'_if-agg-'+str(args.if_agg)
            model_path = './models/t5_'+if_col_header +\
                'nw-'+str(args.max_num_where)+'_if-agg-' + \
                str(args.if_agg)+'_group-g_0/'

    if args.dataset == 'wikisql':
        with open(args.wikisql_path+'/data/train.tables.jsonl') as fp:
            table_list = [json.loads(t) for t in fp.readlines()]
    elif args.dataset == 'wtq':
        table_list = load_wtq_tables(args.group)
    elif args.dataset == 'airlines':
        with open(args.airlines_path+'/airlines_tables_with_types.jsonl') as fp:
            table_list = [json.loads(t) for t in fp.readlines()]
    else:
        print('Dataset name wrong')
    table_dict = {}
    for t in table_list:
        table_dict[t['id']] = t
    dataset_prefix = args.dataset

    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model = model.to(device)

    if args.group != 'all':
        group_id = args.group[2]
    else:
        group_id = args.group

    sql_file = 'data/sql/'+dataset_prefix + split_prefix + '-generated-sql_per-table-' +\
        str(args.num_samples_per_table)+'_group-'+group_id
    if args.part == None:
        sql_file += '.json'
    else:
        sql_file += '_part-' + str(args.part) + '.json'
    with open(sql_file) as fp:
        sql_list = json.load(fp)

    sql_str = []
    for sql_dict in sql_list:
        input_str = convert_sql_to_string(
            sql_dict, table_dict[sql_dict['id']], use_column=args.column_header)
        sql_str.append(input_str)
    inference_data = [[s, str(i)] for i, s in enumerate(sql_str)]

    df = pd.DataFrame(inference_data, columns=['sql', 'question'])
    testing_loader, _ = get_data(df, 1.0)
    predictions, actuals = validate(model, testing_loader)

    output_dict = []
    for i, pred in enumerate(predictions):
        pdict = {}
        pdict['question'] = pred
        idx = int(actuals[i])
        pdict['sql'] = sql_list[idx]
        output_dict.append(pdict)

    if args.model_path == '':
        genq_file = './data/generated_question/' +\
            dataset_prefix + split_prefix + '_gen_quest_g_' + \
            group_id + '_' + if_col_header + args.suffix
    else:
        genq_file = './data/generated_question/' +\
            dataset_prefix + split_prefix + args.model_path
    if args.part == None:
        genq_file += '.json'
    else:
        genq_file += '_part-' + str(args.part) + '.json'
    with open(genq_file, 'w') as fp:
        json.dump(output_dict, fp)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()


    # Dataset and model path arguments
    parser.add_argument('--train_data_path', default='./data/train_qg_t5.csv',type=str)
    parser.add_argument('--test_data_path', default='./data/test_qg_t5.csv',type=str)
    parser.add_argument('--model_path',default='./models/t5_model/',type=str)
    parser.add_argument('--output_dir',default='./models/t5_model/',type=str)
    parser.add_argument('--prediction_file_path', default='./predictions/generated_questions.json',type=str)
    parser.add_argument('--wikisql_path',default='',type=str)
    parser.add_argument('--airlines_path', default='', type=str)
    parser.add_argument('--exp', action='store_true')

    # Arguments used for training and selecting the model
    parser.add_argument('-t', '--task', default='train', type=str)
    parser.add_argument('-g', '--group', default='all', type=str)
    parser.add_argument('-nw', '--max_num_where', default=4, type=int)
    parser.add_argument('-ia', '--if_agg', default=True,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('-col', '--column_header', default=False,
                        type=lambda x: (str(x).lower() in ['true', '1', 'yes']))
    parser.add_argument('-suf', '--suffix', default='', type=str)
    parser.add_argument('-t5', '--t5_path', default='', type=str)
    parser.add_argument('-p', '--part', default=None, type=int)

    # arguments used for generation
    parser.add_argument('-ns', '--num_samples_per_table',
                        default=10, type=int)  # Used for generation
    parser.add_argument('-d', '--dataset', default='wikisql',
                        type=str)  # Used for generation
    parser.add_argument('-s', '--split', default='train',
                        type=str)  # train or dev
    parser.add_argument('-m', '--model_path', default='',
                        type=str)  # train or dev

    args = parser.parse_args()
    print('Arguments =', args)

    if args.column_header:  # input length needs to be bigger if we are addding column names.
        MAX_LEN = 100
    else:
        MAX_LEN = 50

    if args.task == 'train':
        run_train(args)
    elif args.task == 'genq':
        generate_questions_group(args)
