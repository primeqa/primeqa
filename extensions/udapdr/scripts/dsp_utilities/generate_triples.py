

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

###########################################

def generate_triples(reranker_results_filename, chosen_split, chosen_type, chosen_set, given_prompt_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type):

	distillation_triples_filename = "../ColBERT_FM/datasets/distillation_triples_for_ColBERTV2_" 
	if LoTTE_or_BEIR == "LoTTE":
		distillation_triples_filename += str(chosen_split) + "_" + str(chosen_type) + "_" + str(chosen_set) + "_" + str(given_prompt_number) + ".json"
	elif LoTTE_or_BEIR == "BEIR":
		distillation_triples_filename += str(LoTTE_or_BEIR) + "_" + str(chosen_BEIR_set) + "_" + str(chosen_BEIR_type) + "_" + str(given_prompt_number) + ".json"

	with open(reranker_results_filename, 'r') as JSON:
	    reranking_results = json.load(JSON)

	#for key in reranking_results.keys():
	#	print(key)
	#	print(len(reranking_results[key]))
	#	print(type(reranking_results[key]))
		#print(reranking_results[key])

	###########################################

	current_matching_qid = None
	count = 0
	current_retrieved_passages = []
	top_k_check = 5
	triples_7_or_15_or_100 = 100

	qid_to_answer_pid = {}
	qid_to_top_k_passages = {}

	for qid, pid, logit, label in tqdm(zip(reranking_results['qids'], reranking_results['pids'], reranking_results['logits'], reranking_results['labels'])):

		if current_matching_qid == None:
			current_matching_qid = qid

		if label == 1 and count < top_k_check:
			qid_to_answer_pid[qid] = [pid, logit]
		
		elif current_matching_qid != qid:
			qid_to_top_k_passages[current_matching_qid] = current_retrieved_passages
			current_retrieved_passages = []
			count = 0
			current_matching_qid = qid

		current_retrieved_passages.append([pid, logit])
		count += 1

				

	###########################################

	distillation_triples = []
	for qid in qid_to_answer_pid.keys():
		try:
			if triples_7_or_15_or_100 == 7:
				retrieved_pids = qid_to_top_k_passages[qid]
				
				first_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[:7]:
					first_triple.append(pair)
				distillation_triples.append(first_triple)

				second_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[7:14]:
					second_triple.append(pair)
				distillation_triples.append(second_triple)

			elif triples_7_or_15_or_100 == 15:

				retrieved_pids = qid_to_top_k_passages[qid]

				first_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[:15]:
					first_triple.append(pair)
				distillation_triples.append(first_triple)

			elif triples_7_or_15_or_100 == 100:

				retrieved_pids = qid_to_top_k_passages[qid]

				if len(retrieved_pids) < 90:
					print("Error! The pid count of " + str(qid) + " is too small.")
				
				next_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[:15]:
					next_triple.append(pair)
				distillation_triples.append(next_triple)

				next_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[15:30]:
					next_triple.append(pair)
				distillation_triples.append(next_triple)

				next_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[30:45]:
					next_triple.append(pair)
				distillation_triples.append(next_triple)

				next_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[45:60]:
					next_triple.append(pair)
				distillation_triples.append(next_triple)

				next_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[60:75]:
					next_triple.append(pair)
				distillation_triples.append(next_triple)

				next_triple = [qid, qid_to_answer_pid[qid]]
				for pair in retrieved_pids[75:90]:
					next_triple.append(pair)
				distillation_triples.append(next_triple)

		except:
			print("Error with QID: " + str(qid))

	###########################################

	for triple in distillation_triples:
		if triples_7_or_15_or_100 == 7:
			assert len(triple) == 9
		if triples_7_or_15_or_100 == 15 or triples_7_or_15_or_100 == 100:
			assert len(triple) == 17
		else:
			assert False

	output_file = open(distillation_triples_filename, 'w')

	for triple in distillation_triples:
		json.dump(triple, output_file)
		output_file.write("\n")

	print("Saved distillation triples file: " + distillation_triples_filename)
	print("Number of triples created: " + str(len(distillation_triples)))

	return distillation_triples_filename

