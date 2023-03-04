
import pandas as pd
import numpy as np
import json
from sklearn.metrics import ndcg_score
from tqdm import tqdm
import ast

def evaluate_beir(distilled_ranking, chosen_BEIR_set, chosen_BEIR_type):

	distilled_results = pd.read_csv(distilled_ranking, sep="\t", header=None)

	##################################################

	question_id_to_gold_passages = {}
	question_id_to_gold_passage_scores = {}

	with open("../ColBERT_FM/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/qas.jsonl", 'r') as f:
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
				current_reference_score = current_gold_passage_scores[current_index]

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

#evaluate_beir("../zeroshot_results/ColBERTv2_ZeroShot_BEIR_hotpotqa.k=1000.device=gpu.ranking.tsv", "hotpotqa", "test")



