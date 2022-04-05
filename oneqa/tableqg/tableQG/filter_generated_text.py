from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from torch import cuda
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np

device = 'cuda' if cuda.is_available() else 'cpu'
model_name = 'gpt2'

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)

def measure_perplexity(input_text, stride=512):
    # Perplexity code taken from https://huggingface.co/transformers/perplexity.html
    encodings = tokenizer(input_text, return_tensors='pt')

    max_length = model.config.n_positions

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i    # may be different from stride on last loop
        input_ids = encodings.input_ids[:,begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:,:-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)
    return ppl

def append_perplexity_scores(qg_file):
    with open(qg_file) as fp:
        qg_list = json.load(fp)
    for i,q in enumerate(tqdm(qg_list)):
        bestq = q['question'][0]
        try:
            ppl_score = measure_perplexity(bestq)
            qg_list[i]['ppl_score'] = ppl_score.tolist()
        except:
            qg_list[i]['ppl_score'] = np.inf
    return qg_list

def sort_generated_questions_and_save(qg_file):
    qg_list = append_perplexity_scores(qg_file)
    sorted_qg = sorted(qg_list, key=lambda qg:qg['ppl_score'], reverse=False)

    ppl_scored_file = qg_file.replace('.json','_ppl_score.json')
    with open(ppl_scored_file,'w') as fp:
        json.dump(sorted_qg, fp)
    return ppl_scored_file

def question_quality_check(question):
    wh_list = ['what', 'when', 'how', 'where', 'why', 'name', 'get', 'which', 'tell', 'who']
    check = False
    for wh in wh_list:
        if wh in question.lower():
            check = True
    return check

def filter_good_questions(qg_file):
    with open(qg_file) as fp:
        qg = json.load(fp)
    good_qg = []
    for q in qg:
        if question_quality_check(q['question'][0]):
            good_qg.append(q)
    good_qg_file = qg_file.replace('generated_question', 'cleaned_generated_question')
    with open(good_qg_file, 'w') as fp:
        json.dump(good_qg, fp)
    print(good_qg_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--qg_file',default='', type=str)
    args = parser.parse_args()
    sort_generated_questions_and_save(args.qg_file)
