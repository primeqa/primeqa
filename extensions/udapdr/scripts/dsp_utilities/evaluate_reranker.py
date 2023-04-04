
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer, T5ForConditionalGeneration

import pandas as pd
import numpy as np
import ast
import datasets
from transformers import TrainingArguments, Trainer

import pyarrow as pa
import pyarrow.dataset as ds

from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import get_scheduler

import torch
from tqdm.auto import tqdm
import statistics
import time

import subprocess as sp
import os

import json
import random

from sklearn.metrics import ndcg_score
from dsp_utilities.evaluate_beir import evaluate_beir

######################################################################

random_state = 43

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

def evaluate_reranker(reranker_checkpoint_path, chosen_split, chosen_type, chosen_set, device, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type):

	class CustomBERTModel(nn.Module):
	    def __init__(self, model_choice):

	            super(CustomBERTModel, self).__init__()
	          
	            model_encoding = AutoModel.from_pretrained(model_choice)
	            self.embedding_size = 1024
	            self.encoderModel = model_encoding

	            self.first_classifier = nn.Sequential(
	          											nn.Linear(in_features=self.embedding_size, out_features=2)
	          									     )

	            ######################################################################



	    def forward(self, input_ids, attention_mask, overlapping_passage_counts=None):

	        total_output = self.encoderModel(input_ids=input_ids, attention_mask=attention_mask)
	        
	        CLS_token = total_output['last_hidden_state'][:,0,:].view(-1, self.embedding_size)

	        combined_logits = self.first_classifier(CLS_token)

	        return {'combined_logits': combined_logits}

	def tokenize_function(examples):

	    return tokenizer(examples["original"], padding="max_length", truncation=True)#.input_ids

    ############################################################

	if LoTTE_or_BEIR == "LoTTE":
		if chosen_type == "forum":
			zero_shot_ranking_filename = "zeroshot_results/ColBERTv2_ZeroShot:_" + chosen_split.capitalize() + ".k=1000.device=gpu.ranking.tsv"
		elif chosen_type == "search":
			zero_shot_ranking_filename = "zeroshot_results/ColBERTv2_Zeroshot_Search_" + chosen_split.capitalize() + ".k=1000.device=gpu.ranking.tsv"
	elif LoTTE_or_BEIR == "BEIR":
		zero_shot_ranking_filename = "zeroshot_results/ColBERTv2_ZeroShot_BEIR_" + chosen_BEIR_set + ".k=1000.device=gpu.ranking.tsv"

	zero_shot_ranking_results = pd.read_csv(zero_shot_ranking_filename, sep="\t", header=None)

	device = torch.device(device)
	checkpoint_path = reranker_checkpoint_path

	######################################################################

	model_choice = "microsoft/deberta-v3-large"
	chosen_k = 5
	model_max_length = 384
	tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=model_max_length)
	assigned_batch_size = 1
	gradient_accumulation_multiplier = 32

	re_ranking_count = 5
	training_passages_selection_5_20_or_100 = 1

	######################################################

	if chosen_split == "NQ" or chosen_split == "SQuAD":
	    original_queries = pd.read_csv('/dfs/scratch0/okhattab/OpenQA/NQ/dev/questions.tsv', sep="\t", header=None)
	elif LoTTE_or_BEIR == "BEIR":
		original_queries = pd.read_csv('../beir_datasets/' + chosen_BEIR_set + "/" + chosen_BEIR_type + '/questions.tsv', sep="\t", header=None)
	else:
	    original_queries = pd.read_csv('../downloads/lotte/' + chosen_split + '/' + chosen_set + '/questions.' + chosen_type + '.tsv', sep="\t", header=None)

	original_queries.columns = ['qid', 'question']
	original_queries['original_qid'] = original_queries['qid']
	original_queries.set_index('original_qid', inplace=True)
	original_queries.sort_values('original_qid')
	print("original_queries")
	print(original_queries)

	######################################################################

	if chosen_split == "NQ" or chosen_split == "SQuAD":
	    collection = pd.read_csv("/dfs/scratch0/okhattab/OpenQA/collection.tsv", sep="\t")
	    collection.columns = ['pid', 'passage', 'passage_title']
	elif LoTTE_or_BEIR == "BEIR":
	    collection = pd.read_csv("../beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/collection.tsv", sep="\t", header=None)
	    collection.columns = ['pid', 'passage']
	else:
	    collection = pd.read_csv("../downloads/lotte/" + chosen_split + "/" + chosen_set + "/collection.tsv", sep="\t", header=None)
	    collection.columns = ['pid', 'passage']

	collection['original_pid'] = collection['pid']
	collection.set_index('original_pid', inplace=True)
	collection.sort_values('original_pid')
	print("collection")
	print(collection)

	######################################################################

	question_id_to_gold_passages = {}
	question_id_to_gold_passage_scores = {}

	if LoTTE_or_BEIR == "BEIR":
		with open('../beir_datasets/' + chosen_BEIR_set + '/' + chosen_BEIR_type + '/qas.jsonl', 'r') as f:
			qas = f.readlines()
		for line in tqdm(qas):  
			parsed_line = json.loads(line)
			question_id_to_gold_passages[int(parsed_line['qid'])] = parsed_line['answer_pids']
			question_id_to_gold_passage_scores[int(parsed_line['qid'])] = parsed_line['answer_scores']

	else:
		with open('../downloads/lotte/' + chosen_split + '/' + chosen_set + '/qas.' + chosen_type + '.jsonl', 'r') as f:
			qas = f.readlines()

		for line in tqdm(qas):	
			parsed_line = json.loads(line)
			question_id_to_gold_passages[int(parsed_line['qid'])] = parsed_line['answer_pids']

	######################################################################

	original_qid_to_label = {}
	gpt3_qid_to_label = {}

	original_qid_to_earliest_gold_passage_retrieved = {}
	gpt_3_qid_to_earliest_gold_passage_retrieved = {}

	original_qid_to_retrieved_passages = {}
	gpt3_qid_to_retrieved_passages = {}

	original_qid_and_passage_id_to_label = {}
	gpt3_qid_and_passage_id_to_label = {}

	for i in tqdm(range(0, int(len(zero_shot_ranking_results) / 1000))):
	    
	    zero_shot_correct_answer_found_in_top_k = False
	    for j in range(i * 1000, i * 1000 + chosen_k):
	        current_row = zero_shot_ranking_results.iloc[j]
	        if int(current_row[1]) in question_id_to_gold_passages[int(current_row[0])]:
	            zero_shot_correct_answer_found_in_top_k = True
	            if int(current_row[0]) not in original_qid_to_earliest_gold_passage_retrieved:
	                original_qid_to_earliest_gold_passage_retrieved[int(current_row[0])] = j - i * 1000

	            original_qid_and_passage_id_to_label[(int(current_row[1]), j - i * 1000)] = 1

	    original_top_5_passages = [int(zero_shot_ranking_results.iloc[i * 1000 + j][1]) for j in range(0, 5)]
	    original_top_20_passages = [int(zero_shot_ranking_results.iloc[i * 1000 + j][1]) for j in range(0, 20)]
	    original_top_100_passages = [int(zero_shot_ranking_results.iloc[i * 1000 + j][1]) for j in range(0, 100)]

	    original_qid_to_retrieved_passages[int(zero_shot_ranking_results.iloc[j][0])] = [original_top_5_passages, 
	    																				 original_top_20_passages,
	    																				 original_top_100_passages]

	    if zero_shot_correct_answer_found_in_top_k:
	    	original_qid_to_label[int(current_row[0])] = 1
	    else:
	    	original_qid_to_label[int(current_row[0])] = 0

	####################################################

	total_original_query_plus_two_passages = []
	total_original_query_plus_two_passages_labels = []
	total_question_id_matches = []
	total_passage_ids = []

	for i in tqdm(range(0, int(len(zero_shot_ranking_results) / 1000))):

	    current_row = zero_shot_ranking_results.iloc[i * 1000]
	    current_qid = int(current_row[0])

	    current_original_query = original_queries.loc[current_qid]['question']

	    ########################################################################################

	    current_original_retrieved_passages = original_qid_to_retrieved_passages[current_qid]

	    if training_passages_selection_5_20_or_100 == 0:
	    	training_passages_addition = 5
	    elif training_passages_selection_5_20_or_100 == 1:
	    	training_passages_addition = 20
	    else:
	    	training_passages_addition = 100

	    ########################################################################################

	    for j in range(0, training_passages_addition):

	        try:
	        
	        	current_concatenated_queries_plus_passage = current_original_query + " | "
	        	current_concatenated_queries_plus_passage += " ".join(collection.loc[current_original_retrieved_passages[training_passages_selection_5_20_or_100][j]]['passage'].split(" "))
	        	total_original_query_plus_two_passages.append(current_concatenated_queries_plus_passage)

	        	total_question_id_matches.append(current_qid)
	        	total_passage_ids.append(current_original_retrieved_passages[training_passages_selection_5_20_or_100][j])

	        	if current_original_retrieved_passages[training_passages_selection_5_20_or_100][j] in question_id_to_gold_passages[current_qid]:
	        		total_original_query_plus_two_passages_labels.append(1)
	        	else:
	        		total_original_query_plus_two_passages_labels.append(0)

	        except:
	        	print("Got error for " + str(j))

	######################################################

	print("Total Dataset Lengths")
	print(len(total_original_query_plus_two_passages))
	print(len(total_original_query_plus_two_passages_labels))

	total_original_query_plus_two_passages_testing = total_original_query_plus_two_passages[:]

	total_original_query_plus_two_passages_labels_testing = total_original_query_plus_two_passages_labels[:]

	total_question_id_matches_testing = total_question_id_matches[:]

	total_passage_ids_testing = total_passage_ids[:]

	####################################################

	test_dataset_pandas = pd.DataFrame({'combined_label': total_original_query_plus_two_passages_labels_testing, 'original': total_original_query_plus_two_passages_testing,
										'original_qid': total_question_id_matches_testing, "passage_ids": total_passage_ids_testing})
	test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
	test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

	####################################################

	classification_dataset = datasets.DatasetDict({'test' : test_dataset_arrow})
	tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)

	tokenized_datasets = tokenized_datasets.remove_columns(["original"])
	#tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
	tokenized_datasets.set_format("torch")

	####################################################

	eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=assigned_batch_size)

	############################################################

	model = CustomBERTModel(model_choice)

	model.to(device)

	model.load_state_dict(torch.load(checkpoint_path))

	############################################################

	print("Beginning Evaluation")

	#metric = load_metric("accuracy")

	total_original_label_predictions = torch.LongTensor([]).to(device)
	total_original_label_references = torch.LongTensor([]).to(device)

	total_gpt3_label_predictions = torch.LongTensor([]).to(device)
	total_gpt3_label_references = torch.LongTensor([]).to(device)

	total_combined_label_predictions = torch.LongTensor([]).to(device)
	total_combined_label_references = torch.LongTensor([]).to(device)

	total_original_qid_matches = []
	total_passage_ids_of_retrieved_passages = []
	total_positive_label_logit_scores = []

	inference_start = time.time()

	reranking_times = []
	progress_bar = tqdm(range(len(eval_dataloader)))
	for batch in eval_dataloader:

	    with torch.no_grad():

	        start_time = time.time()
	        new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device)} 

	        outputs = model(**new_batch)
	        reranking_times.append(time.time() - start_time)

	        ######################################################################

	        total_original_qid_matches.append(batch['original_qid'].cpu().numpy())
	        total_passage_ids_of_retrieved_passages.append(batch['passage_ids'].cpu().numpy())
	        total_positive_label_logit_scores.append(outputs['combined_logits'][0][1].cpu().numpy())

	        ######################################################################

	        combined_logits = outputs['combined_logits']
	        
	        combined_labels_predictions = torch.argmax(combined_logits, dim=-1)

	        total_combined_label_predictions = torch.cat((total_combined_label_predictions, combined_labels_predictions), 0)
	        total_combined_label_references = torch.cat((total_combined_label_references, batch['combined_label'].to(device)), 0)

	        ######################################################################

	        progress_bar.update(1)

	############################################################

	print("Average Reranking Times: " + str(sum(reranking_times) / len(reranking_times)))

	############################################################

	total_combined_label_predictions = total_combined_label_predictions.tolist()
	total_combined_label_references = total_combined_label_references.tolist()

	total_original_label_references = total_original_label_references.tolist()

	print("Total Positive Labels and Negative labels percentages")
	print(round((total_combined_label_references.count(1) / len(total_combined_label_references)) * 100, 2))
	print(round((total_combined_label_references.count(0) / len(total_combined_label_references)) * 100, 2))

	############################################################

	print("Performing final scoring")

	correct_5 = 0
	correct_20 = 0
	correct_100 = 0
	total = 0

	normal_correct_5 = 0
	normal_correct_20 = 0
	normal_correct_100 = 0
	normal_total = 0

	qid = -1
	retrieved_passages = []
	prediction_scores = [] 

	reference_scores = []
	reranker_scores = []

	for i in range(0, len(total_original_qid_matches)):

		if total_original_qid_matches[i] != qid:

			if qid != -1:
			
				top_5_indices = sorted(range(len(prediction_scores)), key=lambda i: prediction_scores[i])[-re_ranking_count:]

				chosen_5_correct = False
				chosen_20_correct = False
				chosen_100_correct = False
				for index, j in zip(top_5_indices, range(re_ranking_count)):
					if retrieved_passages[int(index)] in question_id_to_gold_passages[qid]:
						if j < 5:
							chosen_5_correct = True
							chosen_20_correct = True 
							chosen_100_correct = True 
						elif j < 20:
							chosen_20_correct = True 
							chosen_100_correct = True
						else:
							chosen_100_correct = True

				##################################

				if chosen_5_correct:
					correct_5 += 1
				if chosen_20_correct:
					correct_20 += 1
				if chosen_100_correct:
					correct_100 += 1
				total += 1

				##################################

				chosen_5_correct = False
				chosen_20_correct = False
				chosen_100_correct = False
				for j in range(0, re_ranking_count):
					if retrieved_passages[j] in question_id_to_gold_passages[qid]:
						if j < 5:
							chosen_5_correct = True
							chosen_20_correct = True 
							chosen_100_correct = True 
						elif j < 20:
							chosen_20_correct = True 
							chosen_100_correct = True
						else:
							chosen_100_correct = True 

				if chosen_5_correct:
					normal_correct_5 += 1
				if chosen_20_correct:
					normal_correct_20 += 1
				if chosen_100_correct:
					normal_correct_100 += 1
				normal_total += 1

				##############################################################

				# nDCG Scores for our approach

				top_10_indices = sorted(range(len(prediction_scores)), key=lambda i: prediction_scores[i])[-10:]
				top_10_passages = [retrieved_passages[int(index)] for index in top_10_indices]
				top_10_scores = [float(prediction_scores[int(index)]) for index in top_10_indices]

				current_references = []
				for passage in top_10_passages:
					if passage in question_id_to_gold_passages[qid]:
						current_index = question_id_to_gold_passages[qid].index(passage)
						if LoTTE_or_BEIR == "BEIR": 
							current_references.append(question_id_to_gold_passage_scores[qid][current_index])
						else:
							current_references.append(1)
					else:
						current_references.append(0)

				reference_scores.append(current_references)
				reranker_scores.append(top_10_scores)

            ##############################################################

			qid = int(total_original_qid_matches[i])
			retrieved_passages = []
			prediction_scores = [] 

		########################################

		retrieved_passages.append(total_passage_ids_of_retrieved_passages[i])
		prediction_scores.append(total_positive_label_logit_scores[i])

	########################################

	if re_ranking_count == 5:
		five_total = total
		five_total_normal = normal_total
	if re_ranking_count == 20:
		five_total = total / 5
		five_total_normal = normal_total / 5

	print("Success@5 Performance of UDAPDR")
	print(round(correct_5 * 100 / five_total, 2))
	print("--------------------------------------")
	print("Success@5 Performance of baseline strategy")
	print(round(normal_correct_5 * 100 / five_total_normal, 2))

	if LoTTE_or_BEIR == "BEIR":

		overall_ncdg_score = ndcg_score(reference_scores, reranker_scores) * 100
		print("Overall NCDG@10 for DSP approach: " + str(overall_ncdg_score))

		print("Zeroshot ColBERTv2")
		evaluate_beir(zero_shot_ranking_filename, chosen_BEIR_set, chosen_BEIR_type)

	reranker_performance = round(correct_5 * 100 / five_total, 2)
	baseline_performance = round(normal_correct_5 * 100 / five_total_normal, 2)

	return reranker_performance, baseline_performance

