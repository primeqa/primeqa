from dataclasses import dataclass
from typing import Dict

from datasets import Dataset
from primeqa.qg.utils.constants import QGSpecialTokens, SqlOperants
from transformers import PreTrainedTokenizer


def _is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

@dataclass
class SqlProcessor():
	"""
	Class for sql dataset processing, like wikisql, contains methods to preprocess data execute sql etc.
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
	def _execute_sql(sql, table):
		"""_summary_

		Args:
			sql (Dict): SQL as a dict provided in WikiSQL dataset. 
			table (Dict): Values in table as a list of list in a dict with "rows" as the key

		Returns:
			List: output of SQL execution. (Answer to the question captured by SQL)
		"""
		rows = table['rows']
		conds = sql['conds']
		num_conds = len(conds['column_index'])
		agg_op = SqlOperants.agg_ops[sql['agg']]

		# rows which pass through the conditions
		selected_cells = []
		for i,row in enumerate(rows):
			cond_passes = True
			for cond_id in range(num_conds):
				col_id = conds['column_index'][cond_id]
				op_id = conds['operator_index'][cond_id]
				const_string = conds['condition'][cond_id]
				if SqlOperants.cond_ops[op_id] == '=':
					if str(row[col_id]) != str(const_string):
						cond_passes = False
				elif SqlOperants.cond_ops[op_id] == '>':
					if not _is_number(row[col_id]) or not _is_number(const_string):
						cond_passes = False
					elif float(row[col_id]) <= float(const_string):
						cond_passes = False
				elif SqlOperants.cond_ops[op_id] == '<':
					if not _is_number(row[col_id]) or not _is_number(const_string):
						cond_passes = False
					elif float(row[col_id]) >= float(const_string):
						cond_passes = False

			if cond_passes:
				selected_cells.append(rows[i][sql['sel']])
		
		if table['types'][sql['sel']] == 'real':
			selected_cells = [float(str(s).replace(',','')) for s in selected_cells]
		else:
			selected_cells = [s.lower() for s in selected_cells]
		
		# SQL might return an empty set. Applying math operations on empty set will break the code
		if selected_cells == []:
			return ['']

		if agg_op == 'select':
			answer =  selected_cells
		elif agg_op == 'maximum':
			answer = [max(selected_cells)]
		elif agg_op == 'minimum':
			answer = [min(selected_cells)]
		elif agg_op == 'count':
			answer = [len(selected_cells)]
		elif agg_op == 'sum':
			answer = [sum(selected_cells)]
		elif agg_op == 'average':
			answer = [sum(selected_cells)/len(selected_cells)]

		return answer

	@staticmethod
	def _create_sql_string(sql, table, answer):
		"""Convert sql in dict format (the way WikiSQL dataset provides) and convert it to string appending
		answer and column names of tables. Appropriate delimiters are used to demarcate differents parts of SQL

		Args:
			sql (_type_): _description_
			table (_type_): _description_
			answer (_type_): _description_

		Returns:
			str: _description_
		"""
		tokens = QGSpecialTokens
		conds = sql['conds']
		num_conds = len(conds['column_index'])

		agg_str = SqlOperants.agg_ops[sql['agg']]
		col_str = table['header'][sql['sel']].replace('/', ' ')

		conds_str_list = []
		for i in range(num_conds):
			where_col = table['header'][conds['column_index'][i]].replace('/', ' ')
			where_op = SqlOperants.cond_ops_string[conds['operator_index'][i]]
			where_const = conds['condition'][i].replace('/', ' ')
			conds_str_list.append((' '+tokens.cond+' ').join([where_col, where_op, where_const]))
		conds_str = (' '+tokens.sep+' ').join(conds_str_list)

		ans_str = str(answer)

		header_str = (' '+tokens.hsep+' ').join(table['header'])

		sql_str = agg_str + ' '+tokens.sep+' ' + col_str + ' '+tokens.sep+' ' + conds_str + \
				' '+tokens.ans+' ' + ans_str + ' '+tokens.header+' ' + header_str
		return sql_str

	@staticmethod
	def preprocess_data(example_batch: Dict):
		"""_summary_

		Args:
			example_batch (Dict): _description_
		"""
		processed_data_dict = {'label': [], 'input': []}

		for sql, table, question in zip(example_batch['sql'], example_batch['table'], example_batch['question']):
			answer = SqlProcessor._execute_sql(sql, table)[0]
			sql_str = SqlProcessor._create_sql_string(sql, table, answer)
			
			processed_data_dict['label'].append(question)
			processed_data_dict['input'].append(sql_str)
		return processed_data_dict

	def load_tables(self):
		raise NotImplementedError('Will implement for V2')
