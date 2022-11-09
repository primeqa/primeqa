import json
import sys

def convert(input_path,output_path):
    data = json.load(open(input_path))
    new_data = []
    for d in data:
        new_data.append({'question_id':d['question_id'],'pred':d['pred']})
    json.dump(new_data,open(output_path,"w"))



if __name__=="__main__":
    convert(sys.argv[1],sys.argv[2])