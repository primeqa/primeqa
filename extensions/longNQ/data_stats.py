# load tydi json and compute stats on the questions and answers
# this works for any data in Kilt-ELI5/LongNQ format
import glob
import gzip
import json
import spacy

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('sentencizer')

from rouge_score import rouge_scorer

rouge = rouge_scorer.RougeScorer(rouge_types=['rougeLsum'], split_summaries=True)


def load_json_from_file(gt_file_patterns, count=None):
    data = []
    if gt_file_patterns.endswith('gz'):
        f = gzip.open(gt_file_patterns, 'rt', encoding='utf-8')
    else:
        f = open(gt_file_patterns, 'rt', encoding='utf-8')
    lines = f.readlines()
    if count == None:
        count = len(lines)
    for line in lines[0:count]:
        data.append(json.loads(line))
    return data

from random import sample
import tqdm

def nlp_count(text):
    tokens = [[token.text for token in sent] for sent in nlp(text, disable=['parser', 'tagger', 'ner']).sents]

    token_count = 0

    for sentence in tokens:
        for word in sentence:
            token_count += 1
    return token_count, len(tokens)


def compute_stats(data):

    stats = {}
    stats['q_words'] = 0
    stats['p_words'] = 0
    stats['a_words'] = 0
    stats['a_sentences'] = 0
    stats['a_per_q'] = 0
    stats['s_per_p'] = 0
    stats['passages'] = 0
    stats['unanswerable'] = 0
    stats['first_word'] = {}
    stats['iaa'] = 0
    stats['iaa_count'] = 0

    for i in tqdm.tqdm(range(len(data))):
                       
        example = data[i]
        
        example_id = example['id']
        question = example["input"]

        try:
            q_word = question.split()[0]
        except:
            print(f"no words? {question}")
            q_word = ""
        annotations = example['output']

        if annotations == None:
            continue

        '''
        # words in question
        # words in passage
        # words in answer
        # sentences in answer
        # of answers per q
        # of passages
        '''
        if q_word in stats['first_word']:
            stats['first_word'][q_word]+= 1
        else:
            stats['first_word'][q_word] = 1

        token_count, _ = nlp_count(question)
        stats['q_words'] += token_count

        # if 'passages' in example:
        #     for passage in example['passages']:
        #         token_count, sentence_count = nlp_count(passage['title'] + " " + passage['text'])
        #         stats['p_words'] += token_count
        #         stats['s_per_p'] += sentence_count
        #         stats['passages'] += 1

        for answer in example['output']:
            # unanswerable
            if answer['answer'] == None or answer['answer'] == "":
                stats['unanswerable'] += 1
                continue
            token_count, sentence_count = nlp_count(answer['answer'])
            stats['a_words'] += token_count
            stats['a_sentences'] += sentence_count
            stats['a_per_q'] += 1

        # the inter-annotator agreement as per ASQA paper:
        # that we measure as the mean ROUGE-L F1 score
        # between each pair of annotations for the same question. 
        scores = 0
        count = 0
        for j in range(len(example['output'])):
            # if 'annotator' in example['output'][j]['meta'] and len(example['output'][j]['meta']['annotator']) > 1:
            #     scores += len(example['output'][j]['meta']['annotator'])
            #     count += len(example['output'][j]['meta']['annotator'])
            for k in range(j+1, len(example['output'])):
                # someone skipped
                if example['output'][j]['answer'] == None or example['output'][k]['answer'] == None \
                    or example['output'][j]['answer'] == "" or example['output'][k]['answer'] == "":
                    continue
                count += 1
                scores += rouge.score(example['output'][j]['answer'], example['output'][k]['answer'])['rougeLsum'][2]
        if count == 0:
            continue
        scores = scores/count
        stats['iaa'] += scores
        stats['iaa_count'] += 1

    # print(stats)
    print(f"Queries\t{len(data)}")
    print(f"A per Q\t{stats['a_per_q']/len(data)}")
    print(f"WORDS in Q\t{stats['q_words']/len(data)}")
    if stats['a_per_q'] > 0:
        print(f"WORDS in A\t{stats['a_words']/stats['a_per_q']}")
        print(f"SENTENCES in A\t{stats['a_sentences']/stats['a_per_q']}")
    if stats['passages'] > 0:
        print(f"WORDS in P\t{stats['p_words']/stats['passages']}")
        print(f"S per P\t{stats['s_per_p']/stats['passages']}")
    print(f"Unanswerable\t{stats['unanswerable']}")
    if stats['iaa'] == 0:
        print(f"IAA: NA")
    else:
        print(f"IAA: {stats['iaa']/stats['iaa_count']}")

    aggegrated_stats = [len(data),stats['a_per_q']/len(data),stats['q_words']/len(data),stats['a_words']/stats['a_per_q'] if stats['a_per_q'] > 0 else 0, stats['a_sentences']/stats['a_per_q'] if stats['a_per_q'] > 0 else 0,stats['iaa']/stats['iaa_count'] if stats['iaa'] > 0 else 0,stats['unanswerable']]

    for word in stats['first_word']:
        if stats['first_word'][word] > 10:
            print(f"{word}\t{stats['first_word'][word]}")

    s=[str(i) for i in aggegrated_stats]
    stats_as_string = ','.join(s)
    print(f"{stats_as_string}")

    return aggegrated_stats

def load_data(data_dir, count=None):

    files = glob.glob(data_dir)
    data = []
    for file_n in files:
        data.extend(load_json_from_file(file_n, count))
    return data

import sys
data_dir = sys.argv[1] #"/dccstor/srosent2/generative/external_datasets/**/*.jsonl"
do_glob = sys.argv[2]

if "**" in data_dir:
    recursive=True
else:
    recursive=False

if do_glob == 'yes':
    all_data_files = glob.glob(data_dir, recursive=recursive)
else:
    all_data_files = [data_dir]

#all_data_files = glob.glob(data_dir, recursive=True)
# all_data_files = glob.glob("/dccstor/srosent2/generative/appen/final/longNQ/*/*.jsonl")
#all_data_files.extend(glob.glob("/dccstor/srosent2/primeqa-mengxia/data/asqa/formatted/ASQA_*.json"))
# all_data_files = ["/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_withkg_best_all/eli5-train*"]
# all_data_files.append(glob.glob("/dccstor/srosent2/primeqa-mengxia/data/dpr-100passages_withkg_best_all/eli5-dev*"))
all_stats = {}

for data_file in all_data_files:
    if "original" in data_file:
        continue
    print(data_file)
    all_stats[data_file[len("/dccstor/srosent2/generative/external_datasets/"):]] = compute_stats(load_data(data_file))

print("Queries,A per Q,WORDS in Q,WORDS in A,SENTENCES in A, IAA, UNANSWERABLE")
for stat in all_stats:
    s=[str(i) for i in all_stats[stat]]
    stats_as_string = ','.join(s)
    print(f"{stat},{stats_as_string}")


# /dccstor/srosent1/miniconda3/envs/primeqa_new/bin/python /dccstor/srosent2/primeqa/primeqa/extensions/longNQ/data_stats.py