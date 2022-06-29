from tqdm import tqdm
from datasets import load_dataset
from primeqa.qg.utils.constants import QGSpecialTokens

class TydiQADataset():
	def __init__(self):
		pass

	def preprocess_data_for_qg(self, data_split='train'):
		data = load_dataset('tydiqa', 'secondary_task', split=data_split)
		processed_data_dict = {'question':[], 'input':[]}
		
		for d in tqdm(data):
			input = d['answers']['text'][0] +' '+QGSpecialTokens.sep+' ' + d['context']
			
			processed_data_dict['input'].append(input)
			processed_data_dict['question'].append(d['question'])
		return processed_data_dict
