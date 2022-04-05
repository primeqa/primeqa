import json
import csv
import time
import nltk
import numpy as np
import sys

agg_ops = ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
cond_ops = ['=', '>', '<', 'OP']
cond_ops_string = ['equal', 'greater', 'lesser', 'OP']

def create_qg_data(wikisql_folder_path,answer_path,split='train',max_num_where=4, if_agg = True, group='all', add_columns=True):

	with open(wikisql_folder_path+'/data/'+split+'.jsonl') as fp:
		tq = [json.loads(t) for t in fp.readlines()]
	with open(wikisql_folder_path+'/data/'+split+'.tables.jsonl') as fp:
		table = [json.loads(t) for t in fp.readlines()]
	with open(answer_path) as fp:
		answers = json.load(fp)

	with open(wikisql_folder_path+'category_groups.json') as fp:
		groups2cat = json.load(fp)
	cat2groups = {}
	for grp in groups2cat:
		for cat in groups2cat[grp]:
			cat2groups[cat] = grp
	with open(wikisql_folder_path+'table_topics.json') as fp:
		table2cat = json.load(fp)

	table_dict = {}
	for t in table:
		table_dict[t['id']] = t

	question_generation_list = [['sql','question']]
	for i,q in enumerate(tq):
		sql = q['sql']
		conds = q['sql']['conds']
		if (sql['agg'] == 0 or if_agg) and len(conds) <= max_num_where:
		# if sql['agg'] == 0 and len(conds) == 1: # only taking non-aggregate questions with one where clause
			question = q['question'] + ' </s>'
			if 'Category:'+table2cat[q['table_id']] in cat2groups:
				tab_grp = cat2groups['Category:'+table2cat[q['table_id']]]
			else:
				tab_grp = str(group)
			if group == 'all' or str(group) not in tab_grp:
				table = table_dict[q['table_id']]

				agg_string = agg_ops[sql['agg']]
				sel_col = table['header'][sql['sel']].replace('/', ' ')
				sql_string = agg_string + ' <extra_id_0> ' + sel_col
				for clause in conds:
					sql_string += ' <extra_id_0> '
					where_col = table['header'][clause[0]].replace('/', ' ')
					where_op = cond_ops_string[clause[1]]
					where_const = str(clause[2]).replace('/', ' ')
					sql_string += ' <extra_id_1> '.join([where_col, where_op, where_const])
				sql_string += ' <extra_id_2> ' + str(answers[i][0])

				if add_columns:
					fname_header = 'col-header_'
					sql_string += ' <extra_id_3> ' + ' <extra_id_4> '.join(table['header'])
				else:
					fname_header = ''

				question_generation_list.append([sql_string.lower() + ' </s>', '<extra_id_99> ' + question.lower()])
	filename = './data/'+split+'_qgen_data_'+ fname_header +\
		'nw-' + str(max_num_where)+'_if-agg-'+str(if_agg)+'_group-'+str(group)+'.csv'
	with open(filename, 'w') as fp:
		tsv_writer = csv.writer(fp, delimiter='\t')
		for q in question_generation_list:
			tsv_writer.writerow(q)
	return question_generation_list

def bluescore(pred_list):
	hyp = [p['predictions'][0] for p in pred_list]
	ref = [p['quesion'] for p in pred_list]

	hyp = [h.lower().split(' ') for h in hyp]
	ref = [r.lower().split(' ') for r in ref]

	bluescore = []
	for weights in [(1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1)]:
		score_list = []
		for i in range(len(hyp)):
			score = nltk.translate.bleu_score.sentence_bleu([ref[i]], hyp[i], weights)
			score_list.append(score)
		bluescore.append(np.mean(score_list))
	return bluescore

if __name__ == '__main__':
	wikisql_folder_path = sys.argv[1] #path to wikisql folder
	answer_path = sys.argv[2]
	create_qg_data(wikisql_folder_path,answer_path)
