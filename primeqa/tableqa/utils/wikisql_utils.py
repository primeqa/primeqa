from dataclasses import dataclass
import ast
# constants for SQL in WikiSQL
#TODO:REMOVE WHEN TABLEQG branch is merged to master

@dataclass
class SqlOperants:
	agg_ops = ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
	cond_ops = ['=', '>', '<', 'OP']
	cond_ops_string = ['equal', 'greater', 'lesser', 'OP']

def _is_number(s):
	try:
		float(s)
		return True
	except ValueError:
		return False

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
		
		# if table['types'][sql['sel']] == 'real':
		# 	selected_cells = [float(str(s).replace(',','')) for s in selected_cells]
		# else:
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
			selected_cells = [c.lstrip('0') for c in selected_cells if len(c)>1]
			answer = [sum([int(ast.literal_eval(x.strip().replace(",",""))) for x in selected_cells if not isinstance(x, int)])]
		elif agg_op == 'average':
			print(selected_cells)
			selected_cells = [c.lstrip('0') if len(c)>1 else c for c in selected_cells ]
			print(selected_cells)
			selected_cells = [int(ast.literal_eval(x.strip().replace(",",""))) for x in selected_cells if not isinstance(x, int)]
			answer = [sum(selected_cells)/len(selected_cells)]

		return answer