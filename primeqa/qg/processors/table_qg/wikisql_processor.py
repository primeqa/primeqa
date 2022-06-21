from tqdm import tqdm
from datasets import load_dataset
from primeqa.qg.utils.constants import SqlOperants, QGSpecialTokens

def _is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

class WikiSqlDataset():
	"""
	Class for wikisql dataset, contains methods to preprocess data execute sql etc.
	"""
	def __init__(self):
		pass
	

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
			tokenizer (str, optional): _description_. Defaults to 'T5'.

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

	def preprocess_data_for_qg(self, data_split='train'):
		"""_summary_

		Args:
			data_split (str, optional): _description_. Defaults to 'train'.
		"""
		data = load_dataset('wikisql', split=data_split)
		processed_data_dict = {'question': [], 'input': []}

		for d in tqdm(data):
			answer = self._execute_sql(d['sql'], d['table'])[0]
			sql_str = self._create_sql_string(d['sql'], d['table'], answer)
			
			processed_data_dict['question'].append(d['question'])
			processed_data_dict['input'].append(sql_str)
		return processed_data_dict

	def load_tables(self):
		raise NotImplementedError('Will implement for V2')