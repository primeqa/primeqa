import json
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser

LL = len("?locale=")


def parse_arguments():
    """
    Parses command line arguments.
    @return: The parsed command line arguments.
    @rtype: argparse.Namespace
    """
    parser = ArgumentParser()
    parser.add_argument('-o', '--output', required=True,
                        help='The template output. Languages will be added before the .jsonl extension.')
    parser.add_argument('multifile', type=str, help="The jsonl input file.")
    parser.add_argument('tsv', type=str, help="The tsv mapping file.")
    return parser.parse_args()


def get_lang_map(tsv_file):
    """
    Generates a language mapping from URLs to languages, based on the URLs - assumes the SAP URL format that contains
    the language in the URL.
    @param tsv_file: The path to the TSV (Tab-Separated Values) file containing language mappings.
    @return: A generator that yields tuples of id and language. Each tuple represents a language mapping found in the TSV file.
    """
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
    """
    @param file_map: A dictionary that maps language identifiers to file objects. Each file object is used to write the processed data for a specific language.
    @param multifile: The path to the input file containing the data to be processed.
    @param output: The path to the output file where the processed data will be written.

    @return: None

    This method processes a multifile containing data and writes separate output files for each language found in the data. The file_map parameter is used to map each language identifier
    * to a corresponding file object. After processing each line in the multifile, the data is checked for a language identifier. If the language identifier is found in the file_map, the
    * data is written to the corresponding output file for that language. If the language identifier is not found in the file_map, a message is printed indicating that the language for the
    * document could not be found. Finally, the file objects in the file_map are closed.
    """
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
    """
    @param file_map: A dictionary containing file objects as values.
    @return: None

    This method iterates through the keys of the file_map dictionary and closes each file object associated with each key.
    """
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
