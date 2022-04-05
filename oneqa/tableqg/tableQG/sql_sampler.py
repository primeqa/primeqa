import json
from tableQG.wikisql_lib.dbengine import DBEngine
from tableQG.wikisql_lib.query import Query
import numpy as np
from copy import deepcopy
from tabulate import tabulate
import math
from tqdm import tqdm
import argparse
from tableQG.utils import load_wtq_tables, sql_execution

agg_ops = ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
cond_ops = ['=', '>', '<', 'OP']
cond_ops_string = ['equal', 'greater', 'lesser', 'OP']


def show_table(table):
    print(tabulate(table['rows'], table['header'], tablefmt="grid"))


def _get_inequality_conds(col, num_conditions=5):
    unique_set = np.unique(col)
    conds_list = []

    for val in unique_set:
        greater_id_list = []
        lesser_id_list = []
        for i in range(len(col)):
            if col[i] > val:
                greater_id_list.append(i)
            elif col[i] < val:
                lesser_id_list.append(i)
        if len(lesser_id_list) > 0:
            conds_list.append([str(val), 2, lesser_id_list])
        if len(greater_id_list) > 0:
            conds_list.append([str(val), 1, greater_id_list])

    # Many inequality conditions can be generated for a real column. This makes it computationally expensive
    # later when creating multiple where clauses. We will sample inequalities here for that reason.
    sampled_idx = np.random.choice(len(conds_list), min(
        num_conditions, len(conds_list)), replace=False)
    sampled_conds_list = [conds_list[i] for i in sampled_idx]

    return sampled_conds_list


def _get_column_freq(table, if_ineq=False):
    rows = table['rows']
    types = table['types']

    if if_ineq:
        num_real_cols = len([t for t in types if t == 'real'])
        if num_real_cols > 0:
            num_ineq_conds = max(round(50/num_real_cols), 1)

    # creating column lists
    cols = [[] for _ in range(len(rows[0]))]
    for r in rows:
        for i, cell in enumerate(r):
            cols[i].append(cell)

    # finding unique in each column
    cols_list = []
    for j, col in enumerate(cols):
        cdict = {}
        for i, cell in enumerate(col):
            if cell not in cdict:
                cdict[cell] = [i]
            else:
                cdict[cell].append(i)
        clist = []
        for c in cdict:
            clist.append([c, 0, cdict[c]])
        # adding inequality conditions
        if types[j] == 'real' and if_ineq:
            clist.extend(_get_inequality_conds(col, num_ineq_conds))

        cols_list.append(clist)
    return cols_list


def _check_condition(conds, cols_list):
    all_rows = []
    for c in conds:
        for cell in cols_list[c[0]]:
            if cell[0] == c[2] and cell[1] == c[1]:  # if conditions match
                all_rows.append(cell[2])
    intersection_len = len(set(all_rows[0]).intersection(*all_rows))

    # check if the condition gives non zero number of rows
    if intersection_len == 0:
        return False

    # check if any subset condition can produce same set of rows
    for r in all_rows:
        rows = deepcopy(all_rows)
        rows.remove(r)
        rlen = len(set(rows[0]).intersection(*rows))
        if rlen == intersection_len:
            return False
    return True


def _get_unique_conditions(wlist):
    wdict = {}
    for wc in wlist:
        conds_str = str(sorted([str(c) for c in wc['conds']]))
        wdict[conds_str] = wc
    wlist = []
    for key in wdict:
        wlist.append(wdict[key])
    return wlist


def get_where_clauses(table, num_where=2, if_ineq=False):
    cols_list = _get_column_freq(table, if_ineq)
    where_dict = {}

    where1_list = []
    for i, c in enumerate(cols_list):
        for cell in c:
            wc = {'conds': [[i, cell[1], cell[0]]], 'rows': cell[2]}
            where1_list.append(wc)
    where_dict['nw-1'] = where1_list

    # removing cells which only appear once before going to multiple where
    cc_list = []
    for i in range(len(table['header'])):
        cc = []
        for cell in cols_list[i]:
            if len(cell[2]) > 1:
                cc.append(cell)
        cc_list.append(cc)
    cols_list = cc_list

    if num_where >= 2:
        where2_list = []
        for i in range(len(cols_list)):
            colA = cols_list[i]
            for j in range(i+1, len(cols_list)):
                colB = cols_list[j]
                for ca in colA:
                    for cb in colB:
                        intersection = list(set(ca[2]) & set(cb[2]))
                        conds = [[i, ca[1], ca[0]], [j, cb[1], cb[0]]]
                        # if len(intersection) < len(colA[ca]) and len(intersection) < len(colB[cb]) and len(intersection) > 0:
                        if _check_condition(conds, cols_list):
                            wc = {'conds': [[i, ca[1], ca[0]], [
                                j, cb[1], cb[0]]], 'rows': intersection}
                            where2_list.append(wc)
        where_dict['nw-2'] = where2_list

    if num_where >= 3:
        where3_list = []
        for w2 in where2_list:
            if len(w2['rows']) > 1:

                for i in range(len(cols_list)):
                    if i not in [c[0] for c in w2['conds']]:
                        for cc in cols_list[i]:
                            conds = deepcopy(w2['conds'])
                            conds.append([i, cc[1], cc[0]])
                            if _check_condition(conds, cols_list):
                                intersection = list(
                                    set(w2['rows']) & set(cc[2]))
                                wc = {'conds': conds, 'rows': intersection}
                                where3_list.append(wc)
        where3_list = _get_unique_conditions(where3_list)
        where_dict['nw-3'] = where3_list

    if num_where == 4:
        where4_list = []
        for w3 in where3_list:
            if len(w3['rows']) > 1:

                for i in range(len(cols_list)):
                    if i not in [c[0] for c in w3['conds']]:
                        for cc in cols_list[i]:
                            conds = deepcopy(w3['conds'])
                            conds.append([i, cc[1], cc[0]])
                            if _check_condition(conds, cols_list):
                                intersection = list(
                                    set(w3['rows']) & set(cc[2]))
                                wc = {'conds': conds, 'rows': intersection}
                                where4_list.append(wc)
        where4_list = _get_unique_conditions(where4_list)
        where_dict['nw-4'] = where4_list
    return where_dict


def sample_sql(table, num_sample, num_where, agg_op=0, if_ineq=False, dont_use_col=[]):
    header = table['header']
    types = table['types']

    multiple_where_dict = get_where_clauses(table, num_where, if_ineq)
    where_list = multiple_where_dict['nw-' + str(num_where)]
    real_cols = [i for i in range(len(types)) if types[i] == 'real']

    if_agg = 0
    if agg_op != 0:
        if_agg = 1

    filtered_where_list = []
    for wc in where_list:
        cols_in_where = [c[0] for c in wc['conds']]
        if if_agg and len(wc['rows']) > 1 and len(set(real_cols) - set(cols_in_where)) >= 1:
            filtered_where_list.append(wc)
        elif not if_agg and len(wc['rows']) == 1:
            filtered_where_list.append(wc)

    if len(filtered_where_list) == 0:
        return []
    sample_ids = np.random.choice(
        len(filtered_where_list), num_sample, replace=True)
    sampled_where_list = [filtered_where_list[i] for i in sample_ids]

    if if_agg:
        clist = real_cols
    else:
        clist = list(range(len(header)))

    sql_list = []
    for wc in sampled_where_list:
        try:
            possible_cols = list(set(clist) - set([c[0] for c in wc['conds']]))
            select_column = np.random.choice(possible_cols)
            sql = {'sel': select_column, 'agg': agg_op}
            sql['conds'] = wc['conds']

            answer = sql_execution(wc, select_column, agg_op, table)
            sql_dict = convert_sql_to_t5_format(sql, header, answer)
            sql_dict['table_id'] = table['id']
            sql_list.append(sql_dict)
        except:
            print('\nQuery creation error')
            # print('query=',sql)
            print('table = ', table['id'])
    return sql_list


def convert_sql_to_t5_format(sql, header, answer):
    sql_dict = {}
    sql_dict['col'] = [agg_ops[sql['agg']], header[sql['sel']]]
    conds = []
    for c in sql['conds']:
        cond = [header[c[0]], cond_ops_string[c[1]], str(c[2])]
        conds.append(cond)
    sql_dict['conds'] = conds
    sql_dict['answer'] = str(answer[0])
    return sql_dict


def controlled_sample_sql(table_list, num_samples_per_table=5, agg_prob=[], num_where_prob=[], ineq_prob=0.0):
    if agg_prob == []:
        # agg_prob = [0.6, 0.1, 0.1, 0.0, 0.1, 0.1] #['select', 'maximum', 'minimum', 'count', 'sum', 'average']
        # ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
        agg_prob = [1.0, 0., 0., 0., 0., 0.]
    if num_where_prob == []:
        # num_where_prob = [0.0, 0.45, 0.3, 0.2, 0.05] # [0,1,2,3,4], there is no zero where clause case
        # [0,1,2,3,4], there is no zero where clause case
        num_where_prob = [0.0, 1., 0., 0., 0.]

    sample_batch_size = 5
    sample_size_list = [sample_batch_size] * \
        math.floor(num_samples_per_table/sample_batch_size)
    if num_samples_per_table % sample_batch_size != 0:
        sample_size_list.append(num_samples_per_table % sample_batch_size)

    all_sql_list = []
    # for table in tqdm(table_list):
    for i, table in enumerate(table_list):
        print(i, len(table['rows']), '\n'+str(table['types']))
        for num_samples in sample_size_list:
            agg_op = np.random.choice(len(agg_prob), 1, True, agg_prob)[0]
            num_where = np.random.choice(
                len(num_where_prob), 1, True, num_where_prob)[0]

            # if to use ineq.
            if_ineq = np.random.choice(2, 1, True, [1-ineq_prob, ineq_prob])[0]
            sql_list = sample_sql(table, num_samples,
                                  num_where, agg_op, if_ineq)

            num_trials = 0
            while len(sql_list) < num_samples:
                diff = num_samples - len(sql_list)
                # print('agg =', agg_op, ' and nw=', num_where)
                # print('diff', diff)
                if 'real' not in table['types'] and agg_op != 0:
                    agg_op = 0
                elif num_where > 1:
                    if if_ineq:
                        num_where -= 1
                    else:
                        if_ineq = 1
                elif num_where == 1 and agg_op != 0:
                    if not if_ineq:
                        if_ineq = 1
                    else:
                        agg_op = 0
                elif num_where == 1 and agg_op == 0:
                    agg_op = np.random.choice([1, 2, 4, 5])
                diff_sql_list = sample_sql(
                    table, diff, num_where, agg_op, if_ineq)
                sql_list.extend(diff_sql_list)

                num_trials += 1
                if num_trials > 5:  # if we cant get it in 10 trials, let us skip this instances
                    print('Unsuccessful.')
                    break

            all_sql_list.extend(sql_list)
    return all_sql_list


def load_group_tables(group_id, split='train'):
    table_path = '/dccstor/cssblr/vishwajeet/pytorch_neural_symbolic_machines_latest/' + \
        'data/wikisql/raw_' + str(group_id) + '/' + split + '.tables.jsonl'
    with open(table_path) as fp:
        table_list = [json.loads(t) for t in fp.readlines()]
    return table_list


def load_airlines_tables():
    table_path = '/dccstor/cmv/saneem/nlqTable/irl_git/airlines_dataset/airlines_tables_with_types.jsonl'
    with open(table_path) as fp:
        table_list = [json.loads(t) for t in fp.readlines()]
    return table_list

def load_cleaned_aitqa_tables():
    # We selected 3 tables and cleaned them for Watson Innovations demo
    file_path = '/dccstor/cmv/saneem/nlqTable/irl_git/QG-tableQA/data/aviation/good_tables.json'
    with open(file_path) as fp:
        table_list = json.load(fp)
    return table_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default='wikisql',
                        help='From which dataset the tables are to be taken. Options: "wikisql", "wtq" and "airlines"', type=str)
    parser.add_argument('-g', '--group', default='all',
                        help='Topic shift group to pick. Tables from training set of this group is loaded to create SQL samples. ' +
                        'Options: "g_0", "g_1", "g_2", "g_3", "g_4" and "all"',
                        type=str)
    parser.add_argument('-s', '--split', default='train',
                        help='Data split. If to generate SQLs for train or dev set. Options: "train" and "dev"', type=str)
    parser.add_argument('-ns', '--num_samples_per_table', default=10,
                        help='Number of samples per table to generate', type=int)

    args = parser.parse_args()

    num_samples_per_table = args.num_samples_per_table
    group_id = args.group

    if args.dataset == 'wikisql':
        table_list = load_group_tables(group_id, args.split)
    elif args.dataset == 'wtq':
        table_list = load_wtq_tables(group_id)
    elif args.dataset == 'airlines':
        table_list = load_airlines_tables()
    elif args.dataset == 'aitqa_good':
        table_list = load_cleaned_aitqa_tables()

    sql_list = controlled_sample_sql(table_list, num_samples_per_table)

    with open(f'data/sql/{args.dataset}_{args.split}-generated-sql_per-table-{num_samples_per_table}_group-{group_id}.json', 'w') as fp:
        json.dump(sql_list, fp)
    print('Done')
