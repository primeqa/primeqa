import json
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True,
                        help='The template output. Languages will be added before the .jsonl extension.')
    parser.add_argument('multifile', type=str, help="The jsonl input file.")
    parser.add_argument('tsv', type=str, help="The tsv mapping file.")
    args = parser.parse_args()

    lang_map = {}
    # map = pd.read_csv(args.tsv, sep='\t', header=None)

    ll = len("?locale=")
    seen = {}
    with open(args.tsv, 'r', encoding='utf8') as inp:
        for i, line in enumerate(inp):
            id, url, title, _ = line.strip().split("\t")
            pos = url.find("?locale=")
            if pos>=0:
                lang = url[pos + ll:][0:2]
            else:
                print(f"Warning: could not find the language for {url}")
            lang_map[id] = lang
            seen[lang] = 1


    file = {}
    for l in seen.keys():
        file[l] = open(args.output.replace(".jsonl", f"-{l}.jsonl"), 'w')

    with open(args.multifile, 'r', encoding="utf8") as f:
        for line in tqdm(f, desc="Processing lines: "):
            data = json.loads(line)
            id = data['document_id'].replace(".txt", "")
            if data['document'].find("Sign In ") >= 0:
                continue
            if id in lang_map:
                file[lang_map[id]].write(json.dumps(data, ensure_ascii=False)+"\n")
            else:
                print(f"Could not find the language for document {id}")

    for l in seen.keys():
        file[l].close()