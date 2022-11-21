import json


resource_path = "data/hybridqa/WikiTables-WithLinks"

def load_data(table_id):
    with open('{}/request_tok/{}.json'.format(resource_path, table_id)) as f:
        requested_documents = json.load(f)
    with open('{}/tables_tok/{}.json'.format(resource_path, table_id)) as f:
        table = json.load(f)
    return table,requested_documents


    


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
                







