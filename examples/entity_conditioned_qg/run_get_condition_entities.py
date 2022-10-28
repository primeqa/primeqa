from transformers import BertTokenizer, BertModel, BertConfig
import torch
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import json
from nltk import sent_tokenize
from glob import glob
import argparse

def get_entities(words, tags):
    all_entities = list()
    i = 0
    total_counter = 0
    while i < len(tags):
        total_counter += 1
        if total_counter > 2000:
            raise Exception
        if tags[i] == "O":
            i+= 1
        else:
            if tags[i].startswith("B-"):
                entity_type = tags[i].split("-")[1]
                entity_indices = list()
                entity_indices.append(i)
                i+= 1
                while i < len(tags) and tags[i] != "O" and (not tags[i].startswith("B-")):
                    total_counter += 1
                    if total_counter > 2000:
                        raise Exception
                    if tags[i] == ("I-" + entity_type):
                        entity_indices.append(i)
                        i+=1
                entity_item = dict()
                entity_item["type"] = entity_type
                entity_item["string"] = " ".join(words[entity_indices[0]:entity_indices[-1]+1])

                entity_item["word_indices"] = deepcopy(entity_indices)
                all_entities.append(deepcopy(entity_item))
            else:
                i+=1
    return all_entities

def get_question_generation_data(input_path, output_path):
    FILTER = ["DATE", "MONEY", "TIME", "PERCENT", "QUANTITY", "ORDINAL", "CARDINAL"]
    psg_data = pickle.load(open(input_path, "rb")) 
    model_version = 'bert-base-uncased'
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    
    output_data = list()
    errors = 0
    for psg in tqdm(psg_data):
        try:
            attention_by_positions = dict()
            psg_att = psg["attentions"]
            psg_mappings = psg["mappings"]
            words = psg["tokens"]
            sentence_b_start = len(tokenizer.encode(psg["title"]))
            for i in range(sentence_b_start, len(psg["attentions"])-1):
                token_index = i - sentence_b_start
                word_index = psg_mappings[token_index]
                if word_index not in attention_by_positions:
                    attention_by_positions[word_index] = list()
                attention_by_positions[word_index].append(psg_att[i])

            tags = psg["tags"]
            all_entities = get_entities(words, tags)
            all_entity_attentions = list()
            for entity in all_entities:
                entity_attention_item = deepcopy(entity)
                attention_scores = list()
                for word_index in entity["word_indices"]:
                    attention_scores.extend(attention_by_positions[word_index])
                entity_attention_item["attention_score"] = np.sum(attention_scores)
                all_entity_attentions.append(deepcopy(entity_attention_item))
            if len(all_entity_attentions) > 0:
                output_item = deepcopy(psg)
                output_item["entity_attentions"] = deepcopy(all_entity_attentions)
                output_data.append(deepcopy(output_item))
        except:
            errors += 1
            continue

    final_data = list()
    for item in output_data:
        all_entities = item["entity_attentions"]
        filtered_items = list()
        for entity in all_entities:
            if entity["type"] not in FILTER:
                filtered_items.append(deepcopy(entity))
        if len(filtered_items) > 4:
            filtered_items = sorted(filtered_items, key=lambda item: item["attention_score"])
            output_item = deepcopy(item)
            generate_entities = list()
            output_item["generate_entities"] = list()
            for i in filtered_items:
                if i["string"] not in generate_entities:
                    output_item["generate_entities"].append(deepcopy(i))
                    generate_entities.append(i["string"])
                if len(output_item["generate_entities"]) >= 3:
                    break
            
            assert len(output_item["generate_entities"]) <=3
            final_data.append(deepcopy(output_item))   
    
    print("Num errors: ", errors)
    print(len(final_data))
    json.dump(final_data, open(output_path, "w")) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--input_path', type=str, help="input pickle path")
    parser.add_argument('--output_path', type=str, help="output json path")
    args = parser.parse_args()

    get_question_generation_data(args.input_dir, args.output_dir)