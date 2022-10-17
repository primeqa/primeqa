from primeqa.tableqa.preprocessors.convert_to_sqa_format import parse_question
from primeqa.qg.processors.table_qg.sql_processor import SqlProcessor
from primeqa.qg.models.table_qg.sql_sampler import SimpleSqlSampler

import pandas as pd
import csv
import nlp
from nlp import load_dataset
import os
from primeqa.tableqa.preprocessors.dataset import DatasetProcessor
from pathlib import Path

def preprocess_wikisql(output_dir,dataset,split):
    """Preprocesses wikisql dataset downloaded from huggingface. Converts it to a format accepted by tapas

    Args:
        output_dir (str): Directory path to store converted intermediate data 
        dataset (Dataset): Downloaded wikisql dataset
        split (str): The dataset split whether train or dev

    Returns:
        str,str: Returns path to the output directory and path to the processed dataset
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    tables_path = Path(output_dir+"/tables/")
    tables_path.mkdir(parents=True, exist_ok=True)
    data_path = os.path.join(output_dir,split+".tsv")
    file_to_write = open(data_path,"wt")
    tsv_writer = csv.writer(file_to_write, delimiter='\t')
    tsv_writer.writerow(['id','question','table_file','answer_coordinates','answer_text','float_answer','aggregation_label'])
    for id, d in enumerate(dataset):
        question = d['question']
        qid = "wikisql_"+str(id)
        table = d['table']
        
        sql = d['sql']
        answer_text,table =  get_answer(table,sql)
        answer_text = [str(i) for i in answer_text]
        table_id,processed_table,min_tokens = preprocess_table(table)
        if answer_text==['']:
            continue
        if min_tokens > 150:
            continue
        table_df = pd.DataFrame.from_dict(processed_table)
        parsed_data = parse_question(table_df,question,answer_text)
        table_df.to_csv(os.path.join(output_dir,"tables/"+str(table_id)+".csv"), sep=',')
        answer_coordinates = parsed_data[2]
        if answer_coordinates=="" or answer_coordinates==None or answer_coordinates==[]:
            continue
        float_answer = parsed_data[3]
        aggregation_label = parsed_data[4]
        tsv_writer.writerow([qid,question,"tables/"+str(table_id)+".csv",answer_coordinates,answer_text,float_answer,aggregation_label])
    print("Preprocessing done")
    file_to_write.close()
    return output_dir,data_path


def preprocess_table(table):
    """This method preprocess the table and converts it into format such as {column_header: [list of values for that column in every row ]}

    Args:
        table (Dict): The table dictionary as provided with wikisql dataset from huggingface

    Returns:
        str,Dict,int: Returns the id of the table, a dictionary of processed table in required format, and the minimum number of tokens in the table which is #rows * #columns
    """
    header = table['header']
    id = table['id']
    rows = table['rows']
    min_tokens = len(header)*len(rows)
    table_data = {}
    for i,h in enumerate(header):
        table_data[h] = [r[i] for r in rows]
    return id,table_data,min_tokens


def get_answer(table,sql):
    """ Executes the SQL provided with wikisql dataset on the table and fetches the answer text

    Args:
        table (Dict): Table Dictonary
        sql (str): sql string

    Returns:
        str,Dict: Returns the answer text and the corrected table
    """
    answer_text = None
    table = SimpleSqlSampler.add_column_types(table)
    answer_text= SqlProcessor._execute_sql(sql,table)
    return answer_text,table

def load_data(out_dir,tokenizer,subset_train=-1,subset_dev=-1):
    """Main function which downloads the wikisql data from huggingface, converts it into required format and preprocessed it and returns the Dataset objects.

    Args:
        out_dir (str): Output directory to store intemediate converted data
        tokenizer (TapasTokenizer): Tokenizer to tokenize the data and get encodings

    Returns:
        Dataset,Dataset: Returns processed train and dev Dataset objects
    """
    print("Preprocessing wikisql dataset")
    dataset_dev = load_dataset('wikisql', split=nlp.Split.VALIDATION)
    dataset_train = load_dataset('wikisql', split=nlp.Split.TRAIN)
    if(subset_dev>-1):
        dataset_dev=dataset_dev.select(range(subset_dev))
    if(subset_train>-1):
        dataset_train=dataset_train.select(range(subset_train))
    root_dir,train_data_path = preprocess_wikisql(out_dir,dataset_train,"train")
    root_dir,dev_data_path  = preprocess_wikisql(out_dir,dataset_dev,"dev")
    dev_data = pd.read_csv(dev_data_path, sep='\t')
    dev_dataset = DatasetProcessor(dev_data, tokenizer,root_dir)
    train_data = pd.read_csv(train_data_path, sep='\t')
    train_dataset = DatasetProcessor(train_data, tokenizer,root_dir)
    return train_dataset,dev_dataset


