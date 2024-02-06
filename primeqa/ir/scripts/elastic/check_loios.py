import json

from argparse import ArgumentParser
from primeqa.ir.scripts.elastic_ingestion import process_url

parser = ArgumentParser()
parser.add_argument("json_docfile", type=str, help="The input jsonl file")
parser.add_argument("loio_file", type=str, help="The loio file")

args = parser.parse_args()

urls = {}

with open(args.json_docfile, "r") as jsonfile:
    for line in jsonfile:
        data = json.loads(line)
        url = data['document_url']
        url, parts = process_url(url, 'sap')
        urls[parts[-1]] = 1

num_missing = 0
with open(args.loio_file, "r") as loios:
    for line in loios:
        line = line.strip()
        if line.find("\t") >= 0:
            loio, count = line.split("\t")
        else:
            loio = line
        if loio not in urls:
            num_missing += 1
            print(f"Missing key {loio}")