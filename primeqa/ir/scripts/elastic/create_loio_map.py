import json
from argparse import ArgumentParser
from primeqa.ir.scripts.elastic_ingestion import process_url

parser = ArgumentParser(description="Python script example.")
parser.add_argument("--output",
                    type=str,
                    help="Specify the output file.")
parser.add_argument("json", type=str,
                    help="The input json file")

args = parser.parse_args()

with open(args.output, "w") as outfile, open(args.json) as inp:
    for line in inp:
        data = json.loads(line)
        if 'document_url' in data:
            url_string = 'document_url'
        elif 'url' in data:
            url_string = 'url'
        else:
            print("Neither 'url' nor 'document_url' in the input json.")

        url, fields = process_url(data[url_string], 'sap')
        print(f"{data['document_id']}\t{fields[-1]}", file=outfile)