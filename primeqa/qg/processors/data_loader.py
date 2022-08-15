from datasets import load_dataset

from .passage_qg.qg_processor import QGProcessor
from primeqa.qg.processors.table_qg.sql_processor import SqlProcessor


class QGDataLoader():
	def __init__(self, 
				tokenizer,
				modality,
				input_max_len,
				target_max_len,
				dataset_name=None,
				):

		self.dataset_name = dataset_name
		# TODO might need to convert data to common format first
		if modality == 'table':
			self.processor = SqlProcessor(tokenizer, input_max_len, target_max_len)
		elif modality == 'passage':
			self.processor = QGProcessor(tokenizer, input_max_len, target_max_len)

	def create(self, dataset=None, data_split='train', data_config=None):
		if dataset is None:
			assert self.dataset_name is not None
			dataset = load_dataset(self.dataset_name, name=data_config, split=data_split)
		return self.processor(dataset)
