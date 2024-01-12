import json
from argparse import ArgumentParser
from primeqa.ir.scripts.elastic_ingestion import process_url

parser = ArgumentParser(description="Create the file2url tsv file from a json document file.")
parser.add_argument("-o", "--output",
                    type=str,
                    help="Specify the output file.")
parser.add_argument("json", type=str,
                    help="The input json file")

args = parser.parse_args()

with open(args.output, "w") as outfile, open(args.json) as inp:
    for line in inp:
        data = json.loads(line)
        url, fields = process_url(data['document_url'], 'sap')
        # print(f"{data['document_id']}\t{fields[-1]}", file=outfile)
        print(f"{data['document_id']}\t{data['document_url']}\t{data['title']}", file=outfile)