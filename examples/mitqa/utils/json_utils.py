import json
def read_data(file_name):
    with open(file_name, 'r') as fin:
        data = json.load(fin)
    return data