import json
import sys

def convert(input_path,output_path):
    """
    The convert function takes in two arguments:
        input_path (str): The path to the file containing the original data.
        output_path (str): The path to write the converted data.
    
    Args:
        input_path: Specify the path of the input file
        output_path: Specify the path of the output file
    
    Returns:
        A json file with the question_id and pred
    """
    data = json.load(open(input_path))
    new_data = []
    for d in data:
        new_data.append({'question_id':d['question_id'],'pred':d['pred']})
    json.dump(new_data,open(output_path,"w"))

def read_data(file_name):
    with open(file_name, 'r') as fin:
        data = json.load(fin)
    return data

if __name__=="__main__":
    convert(sys.argv[1],sys.argv[2])