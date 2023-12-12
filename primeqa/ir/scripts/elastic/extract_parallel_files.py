import json
from primeqa.ir.scripts.elastic_ingestion import read_data
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser(description='Identify parallel data in SAP json files.')
    parser.add_argument('-o', '--output', )
    parser.add_argument('langs', metavar='N', type=str, nargs='*',
                        help='Languages, in the format <lang>:<file>. English needs to be part of the list.' \
                             ' Multiple files are allowed - separated by ","')
    args = parser.parse_args()
    passages = {}
    for filearg in args.langs:
        lang, filepath = filearg.split(':')
        files = filepath.split(',')
        passages[lang] = read_data(files, lang, remote_url=False)

    en_loios = {}
    for i, passage in enumerate(passages['en']):
        en_loios[passage['filePath']] = i

    for lang in passages.keys():
        if lang == 'en':
            continue
