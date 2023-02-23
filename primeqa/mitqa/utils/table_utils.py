import json
import os

resource_path = "data/hybridqa/WikiTables-WithLinks"
resource_path_ottqa = "data/ottqa/all_plain_tables.json"

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
    """
    The fetch_ottqa_passages function takes in a dictionary of the OTTQA dataset and returns two lists:
        1. A list of passages for each row in the table
        2. The corresponding table itself
    
    Args:
        d: Store the table and row_passage_links
        passages_dict: Store the passages that have already been fetched
    
    Returns:
        A tuple of two lists (table and passage rows)
    """
    table = d['table']
    row_passage_links = d['row_passage_links']
    passages_rows = []
    for row_links in row_passage_links:
        row_passage = [passages_dict[i] for i in row_links if i in passages_dict.keys()]
        passages_rows.append(row_passage)
    return table, passages_rows
        


def fetch_table(table_id):
    """
    The fetch_table function takes a table_id as input and returns the table with requested documents.
    The function first loads the data from the json file, then adds passages to each cell in 
    the table based on which document is requested. The function also returns a list of all 
    requested documents.
    
    Args:
        table_id: Specify which table to load
    
    Returns:
        A table
    """
    table,requested_documents = load_data(table_id)
    
    return add_passage_to_cell(table,requested_documents)

def add_passage_to_cell(table,requested_documents):
    """
    The add_passage_to_cell function takes a table and adds passages to each cell.
    The function returns the modified table.
    
    Args:
        table: Get the url of the table
        requested_documents: Retrieve the passages from the documents that are linked to a cell
    
    Returns:
        A dictionary with passages attached to table cells
    """
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
                







