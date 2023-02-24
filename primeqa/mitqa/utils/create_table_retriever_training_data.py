import json
import random
import csv
from tqdm import tqdm

import sys
def linearize_row(row):
    """
    The linearize_row function takes a row of the dataframe and returns a string
    representation of that row. The string representation is just each column name
    and its value concatenated with &quot;is&quot; and &quot;.&quot; between each pair. For example, if 
    the input was {&quot;a&quot;: 1, &quot;b&quot;: 2}, the output would be:
    
    Args:
        row: row of the table
    
    Returns:
        A string of the form: headr is value
    """
    row_str = ""
    for c,r in row.items():
        row_str+=str(c)+" is "+str(r)+" . "
    return row_str

def linearize_table(table_id,all_tables):
    """
    The linearize_table function takes a table ID and the dictionary of all tables,
    and returns a string representing the linearized form of that table.
    
    
    Args:
        table_id: Retrieve the table from the all_tables dictionary
        all_tables: Retrieve the table from the dictionary
    
    Returns:
        A string representation of the table
    """
    table = all_tables[table_id]
    title = table['title']  
    header = table['header']
    table_rows = []
    linearized_table = str(title)+" "+"[HEAD] "+(' | ').join(header)
    for r in table['data']:
        linearized_table+=" [ROW] "+(' | ').join(r)
    return linearized_table
        

def load_all_tables():
    """
    The load_all_tables function loads all of the tables from the OTTQA dataset into a list.
    Each element in this list is a dictionary representing one table, with keys for 'id', 
    'table_number', 'table_html' and 'table_json'. The table's id is its filename (without extension).
    
    
    Args:
    
    Returns:
        A list of dictionaries
    """
    data = json.load(open("data/ottqa/all_plain_tables.json"))
    return data  

def create_collections():
    """
    The create_collections function creates a tsv file with the following columns:
        - id: The unique identifier for this table.
        - text: The linearized version of the table, where each row is on its own line.
        - title: The title of the Wikipedia page containing this table.
    
    Args:
    
    Returns:
        A tsv file with the id, text and title of each table
    """
    data = load_all_tables()
    tsv_file = open("data/ottqa/linearized_tables.tsv", 'w', encoding='utf8', newline='')
    tsv_writer = csv.writer(tsv_file, delimiter='\t',lineterminator='\n')
    for k,v in data.items():
        id = k
        text = linearize_table(k,data)
        title = v['title']
        tsv_writer.writerow([id,text,title])
        
        
    

def gen_triples_from_dict(split):
    """
    The gen_triples_from_dict function generates a TSV file containing the question, positive table, and negative table for each triple in the triples_dict.json file.
    The TSV file is named &quot;triples_&lt;split&gt;.tsv&quot; where &lt;split&gt; is either 'train', 'dev', or 'test'.
    
    
    Args:
        split: Specify which split of the data to generate triples for
    
    Returns:
        A tsv file with a question, the positive table and the negative table
    """
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

    json.dump(question_table_id_dict,open("data/ottqa/triples_dict_"+split+".json","w"))
    
def generate_table_retriever_data(split):
    """
    The generate_table_retriever_data function generates the following files:
        - pos_neg_table_ids.json: A dictionary mapping each table ID to a list of positive and negative table IDs for that table.
        - triples_{train,dev,test}.csv: A CSV file containing all training examples (positive and negative). Each row contains three columns separated by commas. The first column is the question ID, second column is the query string for that question's example in natural language form, and third column is a comma-separated list of candidate tables corresponding to each word in the query string. 
    
    
    Args:
        split: Determine which split of data to generate
  
    """
    gen_pos_neg_table_id_dict(split)
    gen_triples_from_dict(split)
    create_collections()


if __name__ == '__main__':
    generate_table_retriever_data(sys.argv[1])
    