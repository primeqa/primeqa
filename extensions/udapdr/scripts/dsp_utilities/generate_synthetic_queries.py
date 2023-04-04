
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
tqdm.pandas()

import statistics
import time

import subprocess as sp
import os

import json
import random
from random import randrange

import openai

######################################################################

random_state = 45

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

def generate_synthetic_questions_with_fewshot_FLAN(given_prompt, given_passage, flan_model, flan_tokenizer, device):
    
	given_prompt += "Example 4:\n"
	given_prompt += "Document: " + " ".join(str(given_passage).split(" ")[:256]) #  + "\n"

	input_ids = flan_tokenizer.encode(given_prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)
	if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
		print(input_ids.shape)
		print(input_ids.shape[0])
		print(input_ids.shape[1])
		print("Major error! Sequence length exceeds max length")
		return ""
	outputs = flan_model.generate(
	    input_ids=input_ids,
	    max_length=32,
	    do_sample=True,
	    top_p=0.95,
	    num_return_sequences=1)

	query = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

	########################################################

	if query.lower().find("bad question") != -1:
		bad_question_index = query.lower().find("bad question")
		query = query[:bad_question_index]
	query = query.replace("Good Question", "").replace(": ","")

	########################################################

	return query

def detect_problematic_question(given_question):

	problem_detected = False

	if "speaker" in given_question or "Speaker" in given_question:
		problem_detected = True 
	elif "passage" in given_question or "Passage" in given_question:
		problem_detected = True 
	elif "author" in given_question or "Author" in given_question:
		problem_detected = True 
	elif "the issue" in given_question or "the problem" in given_question:
		problem_detected = True 
	elif len(given_question) == 0:
		problem_detected = True 
	elif len(given_question.replace(" ", "")) == 0:
		problem_detected = True 

	return problem_detected

######################################################################

def generate_synthetic_queries(given_prompt, model_choice, sample_count, chosen_split, chosen_type, chosen_set, chosen_device, given_process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, parallelize=True):

	flan_tokenizer = AutoTokenizer.from_pretrained(model_choice, max_length=2048, truncation=True)
	flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

	######################################################################

	if parallelize:

		chosen_device = "cuda:0"

		gpu_count = 4

		print("Parallelization: loading FLAN on to: " + chosen_device)

		heads_per_gpu = len(flan_model.encoder.block) // gpu_count                                                                                                             
		device_map = {                                                                                                                                                      
		    gpu: list(                                                                                                                                                      
		            range(                                                                                                                                                      
		                0 + (gpu * heads_per_gpu),                                                                                                                              
		                (0 + (gpu * heads_per_gpu)) + heads_per_gpu,                                                                                                            
		            )                                                                                                                                                           
		        )                                                                                                                                                               
		    for gpu in range(gpu_count)                                                                                                                                     
		}                                                                                                                                                                   
		flan_model.parallelize(device_map)

		device = chosen_device
		device = torch.device(device)

	else:

		print("No parallelization: loading FLAN on to: " + chosen_device)

		device = chosen_device
		device = torch.device(device)
		flan_model.to(device)

	######################################################################

	if LoTTE_or_BEIR == "LoTTE":
		queries_filename = "datasets/synthetic_question_for_ColBERTV2_" + str(chosen_device) + "_"  + str(given_process_number) + "_" + model_choice.replace("/","-") + "_" + str(sample_count) + "_" + chosen_split + "_" + chosen_type + "_" + chosen_set + ".tsv"
		qas_file_name = "datasets/synthetic_qas_for_ColBERTv2_" + str(chosen_device) + "_" + str(given_process_number) + "_" + model_choice.replace("/","-") + "_" + str(sample_count) + "_" + chosen_split + "_" + chosen_type + "_" + chosen_set + ".jsonl"
	elif LoTTE_or_BEIR == "BEIR":
		queries_filename = "datasets/synthetic_question_for_ColBERTV2_" + str(chosen_device) + "_"  + str(given_process_number) + "_" + model_choice.replace("/","-") + "_" + str(sample_count) + "_" + chosen_BEIR_set + "_" + chosen_BEIR_type + ".tsv"
		qas_file_name = "datasets/synthetic_qas_for_ColBERTv2_" + str(chosen_device) + "_" + str(given_process_number) + "_" + model_choice.replace("/","-") + "_" + str(sample_count) + "_" + chosen_BEIR_set + "_" + chosen_BEIR_type + ".jsonl"

	if LoTTE_or_BEIR == "LoTTE":
		collection = pd.read_csv("../ColBERT_FM/downloads/lotte/" + chosen_split + "/" + chosen_set + "/collection.tsv", sep="\t", header=None)
	elif LoTTE_or_BEIR == "BEIR":
		collection = pd.read_csv("../ColBERT_FM/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/collection.tsv", sep="\t", header=None)

	collection.columns = ['pid', 'passage']
	collection['original_pid'] = collection['pid']
	collection.set_index('original_pid', inplace=True)
	collection.sort_values('original_pid')

	######################################################################

	question_id_to_gold_passages = {}
	gold_passage_id_to_question_id = {}

	if LoTTE_or_BEIR == "LoTTE":
		with open('../downloads/lotte/' + chosen_split + '/' + chosen_set + '/qas.' + chosen_type + '.jsonl', 'r') as f:
		    qas = f.readlines()
	elif LoTTE_or_BEIR == "BEIR":
		with open('../beir_datasets/' + chosen_BEIR_set + '/' + chosen_BEIR_type + '/qas.jsonl', 'r') as f:
		    qas = f.readlines()

	total_answer_pids = set()
	for line in tqdm(qas):	
	    parsed_line = json.loads(line)
	    question_id_to_gold_passages[int(parsed_line['qid'])] = parsed_line['answer_pids']
	    for answer_pid in parsed_line['answer_pids']:
	    	gold_passage_id_to_question_id[int(answer_pid)] = int(parsed_line['qid'])
	    	total_answer_pids.add(int(answer_pid))

	total_pids = set()
	for i in tqdm(range(len(collection))):
		total_pids.add(collection.iloc[i]['pid'])

	total_non_answer_pids = total_pids - total_answer_pids

	######################################################################

	sub_collection = collection.sample(sample_count, replace=True)

	######################################################################

	sub_collection['synthetic_question'] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_fewshot_FLAN(given_prompt, row['passage'], flan_model, flan_tokenizer, device), axis=1)

	######################################################################

	sub_collection['problematic_query'] = sub_collection.progress_apply(lambda row: detect_problematic_question(row['synthetic_question']), axis=1)

	print("Total Synthetic Queries Generated")
	print(len(sub_collection))

	######################################################################

	print("Creating queries file!")

	queries = []
	for i in range(0, len(sub_collection)):
		current_query = sub_collection.iloc[i]['synthetic_question']
		if sub_collection.iloc[i]['problematic_query'] == False:
			queries.append([i, current_query])

	queries_dataframe = pd.DataFrame(queries)
	queries_dataframe.to_csv(queries_filename, index=False, sep="\t", header=None)

	print("Total Synthetic Queries after filtering")
	print(len(queries))

	print("Saving queries to: " + queries_filename)

	######################################################################

	print("Creating qas file!")

	evaluation_qas = []
	for i in range(0, len(sub_collection)):
		current_query = sub_collection.iloc[i]['synthetic_question']#.split(" | ")[0]
		current_pid = sub_collection.iloc[i]['pid']
		if sub_collection.iloc[i]['problematic_query'] == False:
			evaluation_qas.append({'qid': i, 'query': current_query, "answer_pids": [int(current_pid)]})

	output_file = open(qas_file_name, 'w')

	for triple in evaluation_qas:
		json.dump(triple, output_file)
		output_file.write("\n")

	print("Saved qas file: " + qas_file_name)

	# Clear model from GPU memory
	del flan_model
	torch.cuda.empty_cache()

	return queries_filename, qas_file_name



