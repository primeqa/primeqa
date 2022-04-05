import json
import csv

def format_qg(qg_file):
	# convert questions generated using T5 to WikiSQL format.
	path = 'data/cleaned_generated_question/'
	
	filename = path + qg_file
	category = 'g_' + qg_file.split('_g_')[1][0]
	with open(filename) as fp:
		qg_list = json.load(fp)
	
	question_list = []
	gold_answer_list = []
	for qg in qg_list:
		nf = {}
		nf['category'] = category
		nf['phase'] = 1		
		nf['question'] = qg['question'][0]
		nf['table_id'] = qg['sql']['table_id']
		nf['sql'] = qg['sql']
		nf = json.dumps(nf)
		question_list.append(nf)
		
		gold_answer_list.append([qg['sql']['answer']])
	
	question_str = '\n'.join(question_list)
	with open('data/wikisql_format/' + qg_file.replace('.json','') + '.train.jsonl', 'w') as fp:
		fp.write(question_str)
	
	with open('data/wikisql_format/' + qg_file.replace('.json','') + '.train_gold.json', 'w') as fp:
		json.dump(gold_answer_list, fp)
	

def get_wtq_table_freq(data_path):
	with open(data_path) as fp:
		reader = csv.reader(fp, delimiter='\t')
		header = reader.__next__()
		data = [row for row in reader]
	table_dict = {}
	for d in data:
		if d[2] in table_dict:
			table_dict[d[2]] += 1
		else:
			table_dict[d[2]] = 1
	return table_dict
	
def wtq2tableDict(csv_path):
	table_path = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/'+\
				'data/wikitable/raw_input/WikiTableQuestions/'
	csv_path = csv_path.split('.')[0]
	csv_path += '.tsv'
	tpath = table_path + csv_path
	with open(tpath) as fp:
		reader = csv.reader(fp, delimiter='\t')
		header = reader.__next__()
		rows = [row for row in reader]
	
	# check if all rows has same number of columns:
	for r in rows:
		if len(r) != len(header):
			print(csv_path)
			return {}
	
	table_dict = {}
	table_dict['rows'] = rows
	table_dict['header'] = header
	table_dict['id'] = csv_path
	
	table_dict = add_column_types(table_dict)
	return table_dict

def add_column_types(table):
	# adds a type list to the table dict:
	# type is either 'real' or 'text', the list corresponds to columns in the table

	header = table['header']
	rows = table['rows']

	# identifying column types. Initializing with real
	types = ['real'] * len(header)
	for r in rows:
		for i in range(len(header)):
			x = str(r[i]).replace(',','')
			try:
				_ = float(x)
			except ValueError:
				# if not able to convert string to float in any cell the whole column is 'text'
				types[i] = 'text'
	
	# converting str to float for real columns
	for r in range(len(rows)):
		for i in range(len(header)):
			if types[i] == 'real':
				x = str(rows[r][i]).replace(',','')
				rows[r][i] = float(x)

	table['types'] = types
	table['rows'] = rows
	return table

def sql_execution(where_clause, select_column, agg_op, table):
	
	selected_cells = []
	for row_id in where_clause['rows']:
		selected_cells.append(table['rows'][row_id][select_column])
	
	if table['types'][select_column] == 'real':
		selected_cells = [float(str(s).replace(',','')) for s in selected_cells]
	else:
		selected_cells = [s.lower() for s in selected_cells]
	
	#agg_op list -> ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
	if agg_op == 0:
		answer =  selected_cells
	elif agg_op == 1:
		answer = [max(selected_cells)]
	elif agg_op == 2:
		answer = [min(selected_cells)]
	elif agg_op == 3:
		answer = [len(selected_cells)]
	elif agg_op == 4:
		answer = [sum(selected_cells)]
	elif agg_op == 5:
		answer = [sum(selected_cells)/len(selected_cells)]
	
	return answer
		
def load_wtq_tables(group_id = 'g_0'):
	all_train_path = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/'+\
					'data/wikitable/raw_input/WikiTableQuestions/data/random-split-1-train.tsv'
	group_train_path = '/dccstor/cmv/saneem/nlqTable/irl_git/neural-symbolic-machines/'+\
					'data/wikitable/raw_input_folder/raw_input-LO_' + group_id +\
					'/WikiTableQuestions/data/random-split-1-train.tsv'
	
	all_train_tables = set(get_wtq_table_freq(all_train_path).keys())
	loo_group_tables = set(get_wtq_table_freq(group_train_path).keys())
	group_tables = list(all_train_tables.difference(loo_group_tables))
	
	table_list = []
	for tpath in group_tables:
		tdict = wtq2tableDict(tpath)
		if tdict != {}:
			table_list.append(tdict)
	return table_list

def convert_sql_to_string(sql_dict, table=[], use_column=False):
	sql_str = 'generate question: ' + str(sql_dict['col'][0]) + ' <extra_id_0> ' + str(sql_dict['col'][1])
	for cond in sql_dict['conds']:
		sql_str += ' <extra_id_0> '
		sql_str += ' <extra_id_1> '.join([str(c) for c in cond])
	sql_str += ' <extra_id_2> ' + str(sql_dict['answer'])
	if use_column:
		table['header'] = [str(h) for h in table['header']]
		sql_str += ' <extra_id_3> ' + ' <extra_id_4> '.join(table['header'])
	sql_str += ' </s>'
	return sql_str

def convert_to_lisp(category):
	# convert data in "genearted_question" format ".examples" format 
	path = 'data/generated_question/'
	
	filename = path + 'wtq_gen_quest_{}.json'.format(category)
	with open(filename) as fp:
		qg_list = json.load(fp)
	
	examples_list = []
	for i,qg in enumerate(qg_list):
		id = 'syn-' + category + '-' + str(i)
		question = qg['question'][0].replace('"',"'")
		answer = str(qg['sql']['answer']).replace('"',"'")
		template = '(example (id {}) (utterance "{}") (context (graph tables.TableKnowledgeGraph {})) (targetValue (list (description "{}"))))'.format(id, question, qg['sql']['table_id'], answer)
		examples_list.append(template)
	lisp_path = 'data/lisp_format/wtq_gen_quest_{}.exmaples'.format(category)
	with open(lisp_path, 'w') as fp:
		fp.write('\n'.join(examples_list))
	return examples_list

def convert_to_lisp_mac(qg_file):
	# convert data in "genearted_question" format ".examples" format 
	path = 'generated_question/'
	category = qg_file.split('quest_')[1][:3]

	filename = path + qg_file
	with open(filename) as fp:
		qg_list = json.load(fp)
	
	examples_list = []
	for i,qg in enumerate(qg_list):
		id = 'syn-' + category + '-' + str(i)
		question = qg['question'][0].replace('"',"'")
		answer = str(qg['sql']['answer']).replace('"',"'")
		template = '(example (id {}) (utterance "{}") (context (graph tables.TableKnowledgeGraph {})) (targetValue (list (description "{}"))))'.format(id, question, qg['sql']['table_id'], answer)
		examples_list.append(template)
	qg_lisp = qg_file.replace('.json','.examples')
	lisp_path = 'lisp_format/' + qg_lisp
	with open(lisp_path, 'w') as fp:
		fp.write('\n'.join(examples_list))
	return examples_list


## Tagging questions and tables using sempre code. This needs to be done if the TaBERT-WTQ model to work on new data
# After tagging the data with sempre, preprocess script needs to be run. Preprocess script is in MAPO code base which
# needs a python2.7 version code.

# https://github.com/percyliang/sempre/blob/master/tables/README.md