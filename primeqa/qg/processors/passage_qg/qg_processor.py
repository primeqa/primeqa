from dataclasses import dataclass
from typing import Dict

from datasets import Dataset
from primeqa.qg.utils.constants import QGSpecialTokens
from transformers import PreTrainedTokenizer


@dataclass
class QGProcessor():
	"""
	Class for qg processing, contains methods to preprocess data execute sql etc.
	"""
	
	tokenizer: PreTrainedTokenizer
	input_max_len: int
	target_max_len: int

	def __call__(self, dataset) -> Dataset:
		processed_dataset = dataset.map(self.preprocess_data, batched=True)
		tokenized_data = processed_dataset.map(self.convert_to_features, batched=True)
		columns = ['input_ids', 'attention_mask', 'target_ids', 'target_attention_mask']
		tokenized_data.set_format(type='torch', columns=columns)
		return tokenized_data

	def convert_to_features(self, example_batch: Dict):
		# TODO explicitly provide truncation/padding strategy
		input_encodings = self.tokenizer.batch_encode_plus(example_batch['input'], 
										pad_to_max_length=True, max_length=self.input_max_len)
		target_encodings = self.tokenizer.batch_encode_plus(example_batch['label'], 
										pad_to_max_length=True, max_length=self.target_max_len)
		encodings = {
			'input_ids': input_encodings['input_ids'], 
			'attention_mask': input_encodings['attention_mask'],
			'target_ids': target_encodings['input_ids'],
			'target_attention_mask': target_encodings['attention_mask']
		}
		return encodings
	
	@staticmethod
	def preprocess_data(example_batch: Dict):
		processed_data_dict = {'label':[], 'input':[]}
		
		for answers, context, question in zip(example_batch['answers'], example_batch['context'], example_batch['question']):
			input = answers['text'][0] +' '+QGSpecialTokens.sep+' ' + context
			
			processed_data_dict['input'].append(input)
			processed_data_dict['label'].append(question)
		return processed_data_dict
