import nltk
import numpy as np
from datasets import load_metric
from nltk.tokenize import sent_tokenize

def bleuscore(pred_list):
	hyp = [p['predictions'][0] for p in pred_list]
	ref = [p['quesion'] for p in pred_list]

	hyp = [h.lower().split(' ') for h in hyp]
	ref = [r.lower().split(' ') for r in ref]

	bleuscore = []
	for weights in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
		score_list = []
		for i in range(len(hyp)):
			score = nltk.translate.bleu_score.sentence_bleu([ref[i]], hyp[i], weights)
			score_list.append(score)
		bleuscore.append(np.mean(score_list))
	return bleuscore

def rouge_metrics(input_tokenizer):
	# Nested functions used to let compute_metrics get access to tokenizer
	rouge_score = load_metric('rouge')
	tokenizer = input_tokenizer

	# rouge metrics taken from: https://huggingface.co/course/chapter7/5
	def compute_metrics(eval_pred):
		predictions, labels = eval_pred
		# Decode generated summaries into text
		decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
		# Replace -100 in the labels as we can't decode them
		labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
		# Decode reference summaries into text
		decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
		# ROUGE expects a newline after each sentence
		decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
		decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
		# Compute ROUGE scores
		result = rouge_score.compute(
			predictions=decoded_preds, references=decoded_labels, use_stemmer=True
		)
		# Extract the median scores
		result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
		
		return {k: round(v, 4) for k, v in result.items()}
	
	return compute_metrics