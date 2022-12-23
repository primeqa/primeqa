import json
import os

resource_path = "/dccstor/cssblr/vishwajeet/git/hybridqa_primeqa/data/hybridqa/WikiTables-WithLinks"

def load_data(table_id):
    with open('{}/request_tok/{}.json'.format(resource_path, table_id)) as f:
        requested_documents = json.load(f)
    with open('{}/tables_tok/{}.json'.format(resource_path, table_id)) as f:
        table = json.load(f)
    return table,requested_documents


def load_passages(data_path_root):
    passages_dict = json.load(open(os.path.join(data_path_root, 'all_passages.json')))
    return passages_dict

def fetch_ottqa_passages(d,passages_dict):
    table = d['table']
    row_passage_links = d['row_passage_links']
    passages_rows = []
    for row_links in row_passage_links:
        row_passage = [passages_dict[i] for i in row_links if i in passages_dict.keys()]
        passages_rows.append(row_passage)
    return table, passages_rows
    

    


def fetch_table(table_id):
    table,requested_documents = load_data(table_id)
    
    return add_passage_to_cell(table,requested_documents)

def add_passage_to_cell(table,requested_documents):
    p_table = {}
    p_table['url'] = table['url']
    new_header = []
    for h in table['header']:
        new_header.append(h[0])
    p_table['header'] = new_header

    table_data = table['data']
    new_table_data_list =[]
    for row_idx, row in enumerate(table_data):
        new_row_data_list = []
        for col_idx, cell in enumerate(row):
            cell_dict = {}
            cell_dict['cell_value'] = cell[0]
            cell_dict['passages']=[]
            if cell[1]:
                passages = []
                for l in cell[1]:
                    passages.append(requested_documents[l])
                cell_dict['passages'] = passages
            new_row_data_list.append(cell_dict)
        new_table_data_list.append(new_row_data_list)
    p_table['data'] = new_table_data_list
    return p_table
                







