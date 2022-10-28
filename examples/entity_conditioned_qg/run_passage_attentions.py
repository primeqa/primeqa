from transformers import BertTokenizer, BertModel, BertConfig
import torch
import pickle
import numpy as np
from tqdm import tqdm
from copy import deepcopy
import json
from nltk import sent_tokenize
import argparse

def format_attention(attention):
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set "
                             "output_attentions=True when initializing your model.")
        squeezed.append(layer_attention.squeeze(0))
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)

def get_attentions_nq_train_psgs_tokenized(input_path, output_path, model_path):
    model_version = 'bert-base-uncased'
    do_lower_case = True
    tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
    config=BertConfig.from_pretrained(model_version, output_attentions=True)

    model=BertModel.from_pretrained(model_path, config=config)
    input_data = json.load(open(input_path, "r"))
    all_attentions = list()

    for item in tqdm(input_data):
        # try:
            attention_item = dict()
            sentence_a = item["title"]
            sentence_b = item["text"]
            attention_item = deepcopy(item)

            tokens = item["tokens"]
            input_ids = list()
            input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.cls_token))
            input_ids.extend(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentence_a)))
            input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))

            token_type_ids = list()
            for _ in range(0, len(input_ids)):
                token_type_ids.append(0)
            
            token_to_word_mappings = list()
            word_index = 0
            for word in tokens:
                wordpieces = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))
                for wordpiece in wordpieces:
                    token_to_word_mappings.append(word_index)
                    input_ids.append(wordpiece)
                    token_type_ids.append(1)
                word_index = word_index + 1
            
            input_ids.append(tokenizer.convert_tokens_to_ids(tokenizer.sep_token))
            token_type_ids.append(1)

            batch_input_ids = torch.LongTensor([input_ids])
            batch_token_type_ids = torch.LongTensor([token_type_ids])

            attention = model(batch_input_ids, token_type_ids=batch_token_type_ids)[-1]
            attn = format_attention(attention)
            cls_attn = torch.transpose(attn[-1,:,0,:], 0, 1)
            cls_attn = torch.mean(cls_attn, 1)
            attention_item["attentions"] = cls_attn.tolist()
            attention_item["mappings"] = deepcopy(token_to_word_mappings)
            all_attentions.append(deepcopy(attention_item))        
        # except:
        #     continue
    
    pickle.dump(all_attentions, open(output_path, "wb" ) )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser')
    parser.add_argument('--input_path', type=str, help="input json path")
    parser.add_argument('--output_path', type=str, help="output pickle path")
    parser.add_argument('--model_path', type=str, help="model path")
    args = parser.parse_args()

    get_attentions_nq_train_psgs_tokenized(args.input_path, args.output_path, args.model_path)