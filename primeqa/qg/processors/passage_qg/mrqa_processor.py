import random

from tqdm import tqdm
from datasets import load_dataset
from primeqa.qg.utils.constants import QGSpecialTokens

class MRQADataset():
	def __init__(self, dataset_name='SQuAD', dev_set_num=1000):
		self.dataset_name = dataset_name
		self.data = load_dataset('mrqa')
		self.select_and_split_data(dev_set_num)

	def select_and_split_data(self,dev_set_num):
		# We sample a small number of examples from the training as the dev set
		# and use the original dev set as testing set
		training_set = self.data['train'].select(self.data['train'].subset == self.dataset_name)
		testing_set = self.data['dev'].select(self.data['train'].subset == self.dataset_name)
		all_ids = list(range(len(training_set)))
		random.shuffle(all_ids)
		train_ids, dev_ids = all_ids[dev_set_num:], all_ids[:dev_set_num]
		training_dataset = training_set.select(train_ids)
		dev_dataset = training_set.select(dev_ids)
		self.data['train'] = training_dataset
		self.data['dev'] = dev_dataset
		self.data['test']= testing_set


	def preprocess_data_for_qg(self, data_split='train'):
		processed_data_dict = {'question':[], 'input':[]}
		
		for d in tqdm(self.data[data_split]):
			input = d['answers']['text'][0] +' '+QGSpecialTokens.sep+' ' + d['context']
			processed_data_dict['input'].append(input)
			processed_data_dict['question'].append(d['question'])
		return processed_data_dict
