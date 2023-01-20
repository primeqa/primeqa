import numpy as np
from datasets import load_metric


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
                #only single sentence outputs and references expected
		decoded_preds = [pred.strip() for pred in decoded_preds]
		decoded_labels = [label.strip() for label in decoded_labels]
		# Compute ROUGE scores
		result = rouge_score.compute(
			predictions=decoded_preds, references=decoded_labels, use_stemmer=True
		)
		# Extract the median scores
		result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
		
		return {k: round(v, 4) for k, v in result.items()}
	
	return compute_metrics
