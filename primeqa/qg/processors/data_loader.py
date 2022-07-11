from primeqa.qg.processors.passage_qg.tydiqa_processor import TydiQADataset
from datasets import Dataset
from primeqa.qg.processors.passage_qg.squad_processor import SquadDataset
from primeqa.qg.processors.table_qg.wikisql_processor import WikiSqlDataset

class QGDataLoader():
	def __init__(self, 
				tokenizer,
				dataset_name='wikisql', 
				input_max_len=512,
				target_max_len=32
				):
		self.tokenizer = tokenizer
		self.input_max_len = input_max_len
		self.target_max_len = target_max_len

		if dataset_name == 'wikisql':
			self.dataset = WikiSqlDataset()
		elif dataset_name in ['squad', 'squad_v2']:
			self.dataset = SquadDataset(dataset_name)	
		elif dataset_name in ['tydiqa']:	
			self.dataset = TydiQADataset()
		else:
			raise NotImplementedError("this data not supported")
		
	def convert_to_features(self, example_batch):
		input_encodings = self.tokenizer.batch_encode_plus(example_batch['input'], 
										pad_to_max_length=True, max_length=self.input_max_len)
		target_encodings = self.tokenizer.batch_encode_plus(example_batch['question'], 
										pad_to_max_length=True, max_length=self.target_max_len)
		encodings = {
			'input_ids': input_encodings['input_ids'], 
			'attention_mask': input_encodings['attention_mask'],
			'target_ids': target_encodings['input_ids'],
			'target_attention_mask': target_encodings['attention_mask']
		}
		return encodings

	def create(self, data_split='train'):
		processed_data_dict = self.dataset.preprocess_data_for_qg(data_split) # list of dict

		processed_data = Dataset.from_dict(processed_data_dict)
		tokenized_data =  processed_data.map(self.convert_to_features, batched=True)
		columns = ['input_ids', 'attention_mask', 'target_ids', 'target_attention_mask']
		tokenized_data.set_format(type='torch', columns=columns)
		return tokenized_data
