
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel, T5ForConditionalGeneration
from transformers import BertModel, AutoTokenizer, AutoModel, GPT2Tokenizer, T5ForConditionalGeneration

import pandas as pd
import numpy as np
import ast
import datasets
#from datasets import load_metric
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
from dsp_utilities.pytorchtools import EarlyStopping

######################################################################

random_state = 43

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

############################################################

def train_reranker(zeroshot_ranking, synthetic_queries_filename, synthetic_qas_filename, chosen_split, chosen_type, chosen_set, device, given_process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type):

	class CustomBERTModel(nn.Module):
	    def __init__(self, model_choice):

	        super(CustomBERTModel, self).__init__()
	          
	        if model_choice in ["roberta-large", "microsoft/deberta-v3-large"]:
	            self.embedding_size = 1024
	            model_encoding = AutoModel.from_pretrained(model_choice)
	        elif model_choice == "castorini/monot5-base-msmarco-10k":
	            self.embedding_size = 768
	            model_encoding = T5ForConditionalGeneration.from_pretrained(model_choice)
	        elif model_choice == "cross-encoder/ms-marco-MiniLM-L-6-v2":
	            self.embedding_size = 384
	            model_encoding = AutoModel.from_pretrained(model_choice)
	        else:
	            self.embedding_size = 768
	            model_encoding = AutoModel.from_pretrained(model_choice)
	        self.encoderModel = model_encoding

	        ######################################################################

	        self.first_classifier = nn.Sequential(
	          											#nn.Linear(in_features=self.embedding_size, out_features=self.embedding_size),
	          											nn.Linear(in_features=self.embedding_size, out_features=2)
	          									   )


	    def forward(self, input_ids, attention_mask, labels, overlapping_passage_counts=None):

	        total_output = self.encoderModel(input_ids=input_ids, attention_mask=attention_mask)
	        CLS_token = total_output['last_hidden_state'][:,0,:].view(-1, self.embedding_size)

	    	######################################################

	        combined_logits = self.first_classifier(CLS_token)

	        return {'combined_logits': combined_logits}

    ######################################################################

	def tokenize_function(examples):

	    return tokenizer(examples["original"], padding="max_length", truncation=True)#.input_ids



    ############################################################

	zero_shot_ranking_filename = zeroshot_ranking
	zero_shot_ranking_results = pd.read_csv(zero_shot_ranking_filename, sep="\t", header=None)

	device = torch.device(device)

	######################################################################

	model_choice = "microsoft/deberta-v3-large"
	#model_choice = "microsoft/deberta-v3-base"
	selected_model = "Fewshot_FLAN"

	chosen_k = 5
	model_max_length = 384 #512 #384
	tokenizer = AutoTokenizer.from_pretrained(model_choice, model_max_length=model_max_length) #model_max_length=model_max_length
	patience_value = 1
	assigned_batch_size = 1
	eval_batch_size = 16
	gradient_accumulation_multiplier = 32
	chosen_learning_rate = 5e-6
	num_epochs = 1
	num_warmup_steps = 1000 / assigned_batch_size
	include_gpt_3_query = True
	use_original_retrieved_passage = False
	inject_overlap_scores = True

	dev_split_mark = 300000 #200000 #44800 #9800 #185000 #179500 #39440
	re_ranking_count = 5
	training_passages_selection_5_20_or_100 = 2

	perform_validation = False
	cutoff_mark = 100000000

	restrict_questions_for_training = True









	select_random_negatives_from_top_100 = False







	if LoTTE_or_BEIR == "LoTTE":
		reranker_results_filename = '../ColBERT_FM/datasets/reranker_results_for_ColBERTV2_' + str(device) + "_" + str(given_process_number) + "_" + str(re_ranking_count) + "_" + str(training_passages_selection_5_20_or_100) + "_" + selected_model + "_" + chosen_split + "_" + chosen_type + "_" + chosen_set + '.tsv'
		checkpoint_path = "checkpoints/" + str(given_process_number) + "_" + selected_model + "_" + chosen_split + "_" + chosen_type + "_" + str(model_choice.replace("/", "-")) + "_" + str(num_epochs) + "_" + str(include_gpt_3_query) + "_" + str(chosen_learning_rate) + "_" + str(device) + "_" + str(dev_split_mark) + ".pt"
	elif LoTTE_or_BEIR == "BEIR":
		reranker_results_filename = '../ColBERT_FM/datasets/reranker_results_for_ColBERTV2_' + str(device) + "_" + str(given_process_number) + "_" + str(re_ranking_count) + "_" + str(training_passages_selection_5_20_or_100) + "_" + selected_model + "_" + LoTTE_or_BEIR + "_" + chosen_BEIR_set + "_" + chosen_BEIR_type + '.tsv'
		checkpoint_path = "checkpoints/" + str(given_process_number) + "_" + selected_model + "_" + LoTTE_or_BEIR + "_" + chosen_BEIR_set + "_" + str(model_choice.replace("/", "-")) + "_" + str(num_epochs) + "_" + str(include_gpt_3_query) + "_" + str(chosen_learning_rate) + "_" + str(device) + "_" + str(dev_split_mark) + ".pt"

	######################################################################

	original_queries = pd.read_csv(synthetic_queries_filename, sep="\t", header=None)

	original_queries.columns = ['qid', 'question']
	original_queries['original_qid'] = original_queries['qid']
	original_queries.set_index('original_qid', inplace=True)
	original_queries.sort_values('original_qid')
	#print("original_queries")
	#print(original_queries)

	######################################################################

	if chosen_split == "NQ" or chosen_split == "SQuAD":
	    collection = pd.read_csv("/dfs/scratch0/okhattab/OpenQA/collection.tsv", sep="\t")
	    collection.columns = ['pid', 'passage', 'passage_title']
	elif LoTTE_or_BEIR == "BEIR":
	    collection = pd.read_csv("../ColBERT_FM/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/collection.tsv" , sep="\t", header=None)
	    collection.columns = ['pid', 'passage']
	else:
	    collection = pd.read_csv("../ColBERT_FM/downloads/lotte/" + chosen_split + "/" + chosen_set + "/collection.tsv", sep="\t", header=None)
	    collection.columns = ['pid', 'passage']

	collection['original_pid'] = collection['pid']
	collection.set_index('original_pid', inplace=True)
	collection.sort_values('original_pid')
	#print("collection")
	#print(collection)

	######################################################################

	question_id_to_gold_passages = {}

	with open(synthetic_qas_filename, 'r') as f:
	    qas = f.readlines()

	for line in tqdm(qas):	
	    #parsed_line = json.loads(line)
	    parsed_line = ast.literal_eval(line)
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

		current_row = zero_shot_ranking_results.iloc[i * 1000]
		if int(current_row[0]) in question_id_to_gold_passages:
	    
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




	####################################################

	total_original_query_plus_two_passages = []
	total_original_query_plus_two_passages_labels = []
	total_question_id_matches = []
	total_passage_ids = []

	for i in tqdm(range(0, int(len(zero_shot_ranking_results) / 1000))):

	    try:
	    	current_row = zero_shot_ranking_results.iloc[i * 1000]
	    	current_qid = int(current_row[0])
	    	current_original_query = original_queries.loc[current_qid]['question']
	    	found_query = True 
	    except:
	    	found_query = False
	    	print("Current QID not found! " + str(current_qid))

	    if found_query:
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

	    	try:
	    		found_gold_in_zero_shot_results = False 
	    		for j in range(0, training_passages_addition):
	    			if int(collection.loc[current_original_retrieved_passages[training_passages_selection_5_20_or_100][j]]['pid']) in question_id_to_gold_passages[current_qid]:
	    				found_gold_in_zero_shot_results = True
	    	except:
	    		print("Error finding passage: " + str(current_original_retrieved_passages[training_passages_selection_5_20_or_100][j]))

	    	if found_gold_in_zero_shot_results == True or restrict_questions_for_training == False:

	    		if select_random_negatives_from_top_100 == True:

	    			try:

	    				covered_passages = set()

	    				for j in range(0, training_passages_addition):

	    					current_concatenated_queries_plus_passage = current_original_query + " | "
	    					passage_choice = current_original_retrieved_passages[training_passages_selection_5_20_or_100][j]
	    					current_concatenated_queries_plus_passage += " ".join(collection.loc[passage_choice]['passage'].split(" "))

	    					if passage_choice in question_id_to_gold_passages[current_qid]:
	    						
	    						total_original_query_plus_two_passages.append(current_concatenated_queries_plus_passage)
	    						total_question_id_matches.append(current_qid)
	    						total_passage_ids.append(passage_choice)
	    						total_original_query_plus_two_passages_labels.append(1)

	    						covered_passages.add(passage_choice)
	    				
	    				##################################################################

	    				for j in range(0, training_passages_addition - 1):

	    					current_concatenated_queries_plus_passage = current_original_query + " | "
	    					passage_choice = random.choice(current_original_retrieved_passages[2])
	    					while passage_choice in covered_passages: 
	    						passage_choice = random.choice(current_original_retrieved_passages[2])
	    					covered_passages.add(passage_choice)

	    					current_concatenated_queries_plus_passage += " ".join(collection.loc[passage_choice]['passage'].split(" "))

	    					if passage_choice not in question_id_to_gold_passages[current_qid]:
	    						total_original_query_plus_two_passages.append(current_concatenated_queries_plus_passage)
	    						total_question_id_matches.append(current_qid)
	    						total_passage_ids.append(passage_choice)
	    						total_original_query_plus_two_passages_labels.append(0)

	    			except:
	    				print("Error with select_random_negatives_from_top_100")


	    		else:

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

	total_original_query_plus_two_passages_training = total_original_query_plus_two_passages[:dev_split_mark]
	total_original_query_plus_two_passages_testing = total_original_query_plus_two_passages[dev_split_mark:cutoff_mark]

	total_original_query_plus_two_passages_labels_training = total_original_query_plus_two_passages_labels[:dev_split_mark]
	total_original_query_plus_two_passages_labels_testing = total_original_query_plus_two_passages_labels[dev_split_mark:cutoff_mark]

	total_question_id_matches_training = total_question_id_matches[:dev_split_mark]
	total_question_id_matches_testing = total_question_id_matches[dev_split_mark:cutoff_mark]

	total_passage_ids_training = total_passage_ids[:dev_split_mark]
	total_passage_ids_testing = total_passage_ids[dev_split_mark:cutoff_mark]

	####################################################

	training_dataset_pandas = pd.DataFrame({'combined_label': total_original_query_plus_two_passages_labels_training, 'original': total_original_query_plus_two_passages_training,
											'original_qid': total_question_id_matches_training, "passage_ids": total_passage_ids_training})
	training_dataset_arrow = pa.Table.from_pandas(training_dataset_pandas)
	training_dataset_arrow = datasets.Dataset(training_dataset_arrow)

	validation_dataset_pandas = pd.DataFrame({'combined_label': total_original_query_plus_two_passages_labels_testing, 'original': total_original_query_plus_two_passages_testing,
											  'original_qid': total_question_id_matches_testing, "passage_ids": total_passage_ids_testing})
	validation_dataset_arrow = pa.Table.from_pandas(validation_dataset_pandas)
	validation_dataset_arrow = datasets.Dataset(validation_dataset_arrow)

	test_dataset_pandas = pd.DataFrame({'combined_label': total_original_query_plus_two_passages_labels_testing, 'original': total_original_query_plus_two_passages_testing,
										'original_qid': total_question_id_matches_testing, "passage_ids": total_passage_ids_testing})
	test_dataset_arrow = pa.Table.from_pandas(test_dataset_pandas)
	test_dataset_arrow = datasets.Dataset(test_dataset_arrow)

	####################################################

	classification_dataset = datasets.DatasetDict({'train' : training_dataset_arrow, 
	                                        	   'validation': validation_dataset_arrow, 
	                                        	   'test' : test_dataset_arrow})
	tokenized_datasets = classification_dataset.map(tokenize_function, batched=True)


	tokenized_datasets = tokenized_datasets.remove_columns(["original"])
	tokenized_datasets.set_format("torch")

	####################################################

	train_dataloader = DataLoader(tokenized_datasets['train'], batch_size=assigned_batch_size)
	validation_dataloader = DataLoader(tokenized_datasets['validation'], batch_size=assigned_batch_size)
	eval_dataloader = DataLoader(tokenized_datasets['test'], batch_size=eval_batch_size)

	############################################################

	model = CustomBERTModel(model_choice)

	model.to(device)

	############################################################

	criterion = nn.CrossEntropyLoss()
	optimizer = Adam(model.parameters(), lr=chosen_learning_rate)

	num_training_steps = num_epochs * len(train_dataloader)

	lr_scheduler = get_scheduler(
	    name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
	)

	############################################################

	# initialize the early_stopping object
	early_stopping = EarlyStopping(patience=patience_value, verbose=True, path=checkpoint_path)
	print("Checkpoint Path: " + checkpoint_path)

	print("Beginning Training")

	total_epochs_performed = 0

	for epoch in range(num_epochs):

	    train_losses = []
	    valid_losses = []
	    avg_train_losses = []
	    avg_valid_losses = []

	    total_epochs_performed += 1

	    print("Current Epoch: " + str(epoch))

	    progress_bar = tqdm(range(len(train_dataloader)))

	    gradient_accumulation_count = 0
	    train_losses = []
	    valid_losses = []

	    model.train()
	    for batch in train_dataloader:

	        new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device), 
	                     'labels': batch['combined_label'].to(device)} 
	        			 #"overlapping_passage_counts": batch['overlapping_passage_counts'].to(device)}

	        outputs = model(**new_batch)

	        #print("outputs")
	        #print(outputs['combined_logits'])
	        #print(batch['combined_label'])

	        loss = criterion(outputs['combined_logits'], batch['combined_label'].to(device))

	        loss.backward()

	        gradient_accumulation_count += 1
	        if gradient_accumulation_count % (gradient_accumulation_multiplier) == 0:
	            optimizer.step()
	            lr_scheduler.step()
	            optimizer.zero_grad()
	        
	        progress_bar.update(1)
	        train_losses.append(loss.item())

	    ##############

	    if perform_validation:

	        progress_bar = tqdm(range(len(validation_dataloader)))

	        model.eval()
	        for batch in validation_dataloader:

	            with torch.no_grad():

	                new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device), 
	                             'labels': batch['combined_label'].to(device)} 
	            			     #"overlapping_passage_counts": batch['overlapping_passage_counts'].to(device)}

	                outputs = model(**new_batch)

	                loss = criterion(outputs['combined_logits'], batch['combined_label'].to(device))
	                progress_bar.update(1)

	                valid_losses.append(loss.item())

	        print("Epoch #" + str(epoch))
	        train_loss = np.average(train_losses)
	        valid_loss = np.average(valid_losses)
	        avg_train_losses.append(train_loss)
	        avg_valid_losses.append(valid_loss)
	        
	        epoch_len = len(str(num_epochs))
	        
	        print_msg = (f'[{epoch:>{epoch_len}}/{num_epochs:>{epoch_len}}] ' +
	                     f'train_loss: {train_loss:.5f} ' +
	                     f'valid_loss: {valid_loss:.5f}')
	        
	        print(print_msg)
	        
	        # clear lists to track next epoch
	        train_losses = []
	        valid_losses = []
	        
	        # early_stopping needs the validation loss to check if it has decresed, 
	        # and if it has, it will make a checkpoint of the current model
	        early_stopping(valid_loss, model)
	        
	        if early_stopping.early_stop:
	            print("Early stopping")
	            break

	    else:

	        valid_loss = 0
	        early_stopping(valid_loss, model)
	        
	        if early_stopping.early_stop:
	            print("Early stopping")
	            break

	############################################################

	#model.load_state_dict(torch.load(checkpoint_path))

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
	        new_batch = {'input_ids': batch['input_ids'].to(device), 'attention_mask': batch['attention_mask'].to(device), 
	                     'labels': batch['combined_label'].to(device)} 
	        			 #"overlapping_passage_counts": batch['overlapping_passage_counts'].to(device)}

	        outputs = model(**new_batch)
	        reranking_times.append(time.time() - start_time)

	        ######################################################################

	        total_original_qid_matches.append(batch['original_qid'].cpu().numpy())
	        total_passage_ids_of_retrieved_passages.append(batch['passage_ids'].cpu().numpy())
	        #total_positive_label_logit_scores.append(outputs['combined_logits'][0][1].cpu().numpy())
	        for j in range(0, len(outputs['combined_logits'])):
	            total_positive_label_logit_scores.append(outputs['combined_logits'][j][1].cpu().numpy())

	        ######################################################################

	        combined_logits = outputs['combined_logits']
	        
	        combined_labels_predictions = torch.argmax(combined_logits, dim=-1)

	        total_combined_label_predictions = torch.cat((total_combined_label_predictions, combined_labels_predictions), 0)
	        total_combined_label_references = torch.cat((total_combined_label_references, batch['combined_label'].to(device)), 0)

	        ######################################################################

	        progress_bar.update(1)

	############################################################

	#print("Average Reranking Times: " + str(sum(reranking_times) / len(reranking_times)))

	############################################################

	total_combined_label_predictions = total_combined_label_predictions.tolist()
	total_combined_label_references = total_combined_label_references.tolist()

	total_original_label_references = total_original_label_references.tolist()

	#results = metric.compute(references=total_combined_label_references, predictions=total_combined_label_predictions)
	#print("Accuracy for Model Training Task: " + str(results['accuracy']))

	#print("Total Positive Labels and Negative labels percentages")
	#print(round((total_combined_label_references.count(1) / len(total_combined_label_references)) * 100, 2))
	#print(round((total_combined_label_references.count(0) / len(total_combined_label_references)) * 100, 2))

	############################################################

	print("Creating re-ranking results file!")
	reranker_results = {}
	reranker_results['qids'] = [int(qid) for qid_list in total_original_qid_matches for qid in qid_list]
	reranker_results['pids'] = [int(pid) for pid_list in total_passage_ids_of_retrieved_passages for pid in pid_list]
	reranker_results['logits'] = [float(logit) for logit in total_positive_label_logit_scores]
	reranker_results['labels'] = total_combined_label_references

	total_original_qid_matches = [int(qid) for qid_list in total_original_qid_matches for qid in qid_list]
	total_passage_ids_of_retrieved_passages = [int(pid) for pid_list in total_passage_ids_of_retrieved_passages for pid in pid_list]

	# Sanity check
	assert len(reranker_results['qids']) == len(reranker_results['pids'])
	assert len(reranker_results['pids']) == len(reranker_results['logits'])
	assert len(reranker_results['logits']) == len(reranker_results['labels'])

	with open(reranker_results_filename, 'w') as fp:
	    json.dump(reranker_results, fp)

	print("Saved results to: " + str(reranker_results_filename))

	############################################################

	#print("Performing final scoring")

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

	for i in range(0, len(total_original_qid_matches)):

		if total_original_qid_matches[i] != qid:

			if qid != -1:
			
				top_5_indices = sorted(range(len(prediction_scores)), key=lambda i: prediction_scores[i])[-re_ranking_count:]

				chosen_5_correct = False
				chosen_20_correct = False
				chosen_100_correct = False
				for index, j in zip(top_5_indices, range(len(top_5_indices))):
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
				for j in range(0, len(retrieved_passages)):
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

			qid = int(total_original_qid_matches[i])
			retrieved_passages = []
			prediction_scores = [] 

		########################################

		retrieved_passages.append(total_passage_ids_of_retrieved_passages[i])
		prediction_scores.append(total_positive_label_logit_scores[i])

	########################################

	print("Success@5 Performance of our technique")
	print(round(correct_5 * 100 / total, 2))
	print("Success@20 Performance of our technique")
	print(round(correct_20 * 100 / total, 2))
	print("Success@100 Performance of our technique")
	print(round(correct_100 * 100 / total, 2))
	print("--------------------------------------")
	print("Success@5 Performance of normal strategy")
	print(round(normal_correct_5 * 100 / normal_total, 2))
	print("Success@5 Performance of normal strategy")
	print(round(normal_correct_20 * 100 / normal_total, 2))
	print("Success@5 Performance of normal strategy")
	print(round(normal_correct_100 * 100 / normal_total, 2))

	reranker_success_at_five = round(correct_5 * 100 / total, 2)
	baseline_success_at_5 = round(normal_correct_5 * 100 / normal_total, 2)

	return checkpoint_path, reranker_results_filename, reranker_success_at_five, baseline_success_at_5


