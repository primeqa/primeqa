
import pandas as pd
import numpy as np
import json
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import ast

def evaluate_beir(distilled_ranking, chosen_BEIR_set, chosen_BEIR_type, chosen_LoTTE_split=None, chosen_LoTTE_type=None, chosent_LoTTE_set=None):

	distilled_results = pd.read_csv(distilled_ranking, sep="\t", header=None)

	##################################################

	question_id_to_gold_passages = {}
	question_id_to_gold_passage_scores = {}

	if chosen_LoTTE_split != None:

		with open("../downloads/lotte/" + chosen_LoTTE_split + "/" + chosent_LoTTE_set + "/qas." + chosen_LoTTE_type + ".jsonl", 'r') as f:
		    qas = f.readlines()

		for line in qas:	
		    parsed_line = ast.literal_eval(line)
		    question_id_to_gold_passages[int(parsed_line['qid'])] = parsed_line['answer_pids']
		    question_id_to_gold_passage_scores[int(parsed_line['qid'])] = [1 for gold_passage in question_id_to_gold_passages]

	else:

		with open("../beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/qas.jsonl", 'r') as f:
		    qas = f.readlines()

		for line in qas:	
		    parsed_line = ast.literal_eval(line)
		    question_id_to_gold_passages[int(parsed_line['qid'])] = parsed_line['answer_pids']
		    question_id_to_gold_passage_scores[int(parsed_line['qid'])] = parsed_line['answer_scores']

	##################################################

	scores = []
	references_scores = []

	for i in range(0, int(len(distilled_results) / 1000)):

		current_scores = []
		current_reference_scores = []

		for j in range(0, 10):

			current_row = distilled_results.iloc[i * 1000 + j]
			current_gold_passages = question_id_to_gold_passages[int(current_row[0])]
			current_gold_passage_scores = question_id_to_gold_passage_scores[int(current_row[0])]

			current_reference_score = 0
			if int(current_row[1]) in current_gold_passages:
				current_index = current_gold_passages.index(int(current_row[1]))
				if chosen_LoTTE_split == None:
					current_reference_score = current_gold_passage_scores[current_index]
				else:
					current_reference_score = 1

			current_reference_scores.append(current_reference_score)
			current_scores.append(current_row[3])

		scores.append(current_scores)
		references_scores.append(current_reference_scores)

	#########################################

	overall_ncdg_score = ndcg_score(references_scores, scores)
	overall_ncdg_score = overall_ncdg_score * 100

	print("NCDG@10: " + str(overall_ncdg_score))

	return overall_ncdg_score

################################


