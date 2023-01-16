import json
import random
import csv
from tqdm import tqdm

import sys
def linearize_row(row):
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" is "+str(r)+" . "
        # row_str+=str(c)+" "+str(r)+" "
    return row_str

def linearize_table(table_id,all_tables):
    table = all_tables[table_id]
    title = table['title']  
    header = table['header']
    table_rows = []
    linearized_table = str(title)+" "+"[HEAD] "+(' | ').join(header)
    for r in table['data']:
        linearized_table+=" [ROW] "+(' | ').join(r)
    return linearized_table
        

def load_all_tables():
    data = json.load(open("data/ottqa/all_plain_tables.json"))
    return data  

def create_collections():
    data = load_all_tables()
    tsv_file = open("data/ottqa/linearized_tables.tsv", 'w', encoding='utf8', newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t',lineterminator='\n')
    for k,v in data.items():
        id = k
        text = linearize_table(k,data)
        title = v['title']
        tsv_writer.writerow([id,text,title])
        
        
    

def gen_triples_from_dict(split):
    all_tables = load_all_tables()
    triples_data = json.load(open("data/ottqa/triples_dict_"+split+".json"))
    tsv_file = open("data/ottqa/triples_"+split+".tsv", 'w', encoding='utf8', newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
    for k,v in tqdm(triples_data.items()):
        question = k
        pos_table_id = v['pos_table']
        neg_table_id = v['neg_table']
        linearized_pos = linearize_table(pos_table_id,all_tables)
        linearized_neg = linearize_table(neg_table_id,all_tables)
        tsv_writer.writerow([question,linearized_pos,linearized_neg])
    

def gen_pos_neg_table_id_dict(split):
    """Generates training data for training dpr
    """
    data = json.load(open("data/ottqa/released_data/"+split+".json"))
    question_table_id_dict = {}
    all_table_ids = list(load_all_tables().keys())
    neg_tables= all_table_ids.copy()
    not_found = []
    for d in tqdm(data):
        question = d['question']
        pos_table_id = d['table_id']
        if pos_table_id not in all_table_ids:
            not_found.append(pos_table_id)
            continue
        else:
            neg_tables.remove(pos_table_id)
        neg_table_id = random.choice(neg_tables)
        question_table_id_dict[question] = {"pos_table":pos_table_id,"neg_table":neg_table_id}
        neg_tables.append(pos_table_id)

    print(not_found)
    json.dump(question_table_id_dict,open("data/ottqa/triples_dict_"+split+".json","w"))
    
def generate_table_retriever_data(split):
    gen_pos_neg_table_id_dict(split)
    print("Generate pos neg dict")
    gen_triples_from_dict(split)
    create_collections()


if __name__ == '__main__':
    generate_table_retriever_data(sys.argv[1])
    