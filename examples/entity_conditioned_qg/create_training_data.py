import spacy 
import json 
from copy import deepcopy
import random
from tqdm import tqdm
import numpy as np
import argparse
from nltk import sent_tokenize, word_tokenize

def match_with_passage(passage, ent_text):
    for p in passage.noun_chunks:
        if p.text.lower() == ent_text.lower():
            return p.text, (p.start_char, p.end_char)
        elif len(ent_text.strip().split()) > 1 and ent_text.lower() in p.text.lower():
            return p.text, (p.start_char, p.end_char)
        elif len(p.text.strip().split()) > 1 and p.text.lower() in ent_text.lower():
            return p.text, (p.start_char, p.end_char)
    return "", (0,0)

def get_answer_sentence(text, answer, answer_start):
    sents = sent_tokenize(text)
    for sent in sents:
        if text.index(sent) <= answer_start and (text.index(sent) + len(sent)) > answer_start:
            if answer not in sent:
                return ""
            words = word_tokenize(sent)
            if len(words) > 3:
                return words[0] + " ... " + words[-2] + " " + words[-1]
            else:
                return ""    
    return ""

def convert_data(input_data_path, output_data_path):
    # First line is header
    input_data = open(input_data_path).readlines()[1:]
    output_data = list()
    nlp = spacy.load("en_core_web_sm")

    p_counts = 0
    m_counts = 0
    subj_counts = 0
    subj_matches = 0

    examples = list()

    for line in tqdm(input_data):
        item = json.loads(line)
        example_item = dict()
        qitem = item["qas"][0]
        example_item["id"] = qitem["qid"]
        example_item["question"] = qitem["question"]
        example_item["answer"] = qitem["detected_answers"][0]["text"]
        example_item["answer_start"] = qitem["detected_answers"][0]["char_spans"][0][0]
        example_item["context"] = item["context"]
        examples.append(deepcopy(example_item))

    output_writer = open(output_data_path, "w")

    for example in tqdm(examples):
        answer_sentence = get_answer_sentence(example["context"], example["answer"], example["answer_start"])
        if answer_sentence:
            p_counts += 1
            target = answer_sentence + " *=- " + example["answer"] + " @#& " + example["question"]
            question = nlp(example["question"])
            passage = nlp(example["context"])
            
            nsubj = None
            objs = list()
            for chunk in question.noun_chunks:
                if chunk.root.dep_ == "dobj" or chunk.root.dep_ == "pobj":
                    objs.append(chunk.text)
                elif chunk.root.dep_ == "nsubj":
                    if not chunk.text.startswith(example["question"].split()[0].strip()):
                        nsubj = chunk.text

            matches = list()
            for obj in objs:
                m, span= match_with_passage(passage, obj)
                if m != "":
                    matches.append((obj, m, span))

            if nsubj != None:
                subj_counts += 1
                sub_m,span = match_with_passage(passage, nsubj)
                if sub_m != "":
                    subj_matches += 1
                    matches.append((nsubj, sub_m, span))

            if len(matches) > 0:
                a,conditioning_entity,span = matches[0]
                m_counts += 1
                source = example["context"][0:span[0]] + " *=- " + example["context"][span[0]:span[1]] + " @#& " + example["context"][span[1]:]
                output_item = dict()
                output_item["id"] = example["id"]
                output_item["question"] = target
                output_item["context"] = source
                output_item["answers"] = {"text": [conditioning_entity], "answer_start": [span[0]+5]}
                output_writer.write(json.dumps(deepcopy(output_item)) + "\n")


    print("Total: ", len(examples))
    print("Passages: ", p_counts)
    print("Matches: ", m_counts)
    print("Sub counts: ", subj_counts)
    print("Subject matches: ", subj_matches)
    
    output_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--input_path', type=str, help="input jsonl path")
    parser.add_argument('--output_path', type=str, help="output jsonl path")
    args = parser.parse_args()

    convert_data(args.input_path, args.output_path)
