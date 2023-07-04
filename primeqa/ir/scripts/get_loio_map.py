import json
from tqdm import tqdm

with open("output.json") as inp:
    data = json.load(inp)
    for d in tqdm(data):
        url = d['document_url']
        id = d['document_id']
        loio = url.split('/')[-1].replace(".html",'')
        print (f"{id}\t{loio}")