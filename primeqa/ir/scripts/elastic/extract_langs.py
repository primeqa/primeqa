import json
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

LL = len("?locale=")


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True,
                        help='The template output. Languages will be added before the .jsonl extension.')
    parser.add_argument('multifile', type=str, help="The jsonl input file.")
    parser.add_argument('tsv', type=str, help="The tsv mapping file.")
    return parser.parse_args()


def get_lang_map(tsv_file):
    with open(tsv_file, 'r', encoding='utf8') as inp:
        for i, line in enumerate(inp):
            id, url, title, _ = line.strip().split("\t")
            pos = url.find("?locale=")
            if pos >= 0:
                lang = url[pos + LL:][0:2]
            else:
                print(f"Warning: could not find the language for {url}")
            yield id, lang


def process_multifile(file_map, multifile, output):
    for l in file_map.keys():
        file_map[l] = open(output.replace(".jsonl", f"-{l}.jsonl"), 'w')
    with open(multifile, 'r', encoding="utf8") as f:
        for line in tqdm(f, desc="Processing lines"):
            data = json.loads(line)
            id = data['document_id'].replace(".txt", "")
            if data['document'].find("Sign In ") >= 0:
                continue
            if id in file_map:
                file_map[id].write(json.dumps(data, ensure_ascii=False) + "\n")
            else:
                print(f"Could not find the language for document {id}")
    close_files(file_map)


def close_files(file_map):
    for l in file_map.keys():
        file_map[l].close()


def main():
    args = parse_arguments()

    lang_map = {}
    seen = {}
    for id, lang in get_lang_map(args.tsv):
        lang_map[id] = lang
        seen[lang] = 1

    process_multifile(lang_map, args.multifile, args.output)


if __name__ == '__main__':
    main()
