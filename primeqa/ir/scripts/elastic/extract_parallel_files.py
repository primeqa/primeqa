import json, re
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
        passages[lang] = read_data(files, lang, remote_url=False,
                                   data_type="sap", docname2url=None, doc_based=True)

    en_loios = {}
    for i, passage in enumerate(passages['en']):
        en_loios[passage['filePath']] = i

    en_passages = passages['en']
    for lang in passages.keys():
        if lang == 'en':
            continue
        en_output = args.output.replace(".txt", f".en_{lang}.en.jsonl")
        foreign_output = args.output.replace(".txt", f".en_{lang}.{lang}.jsonl")
        with open(en_output, 'w', encoding='utf') as en, \
            open(foreign_output, 'w', encoding='utf') as foreign:
            for p in passages[lang]:
                filePath = p['filePath']
                if filePath in en_loios:
                    en_passage =  en_passages[en_loios[filePath]]
                    en.write(json.dumps({'id':en_passage['id'], 'text':en_passage['text']},
                                        ensure_ascii=False)+"\n")
                    foreign.write(json.dumps({'id':p['id'], 'text':p['text']},
                                             ensure_ascii=False)+"\n")