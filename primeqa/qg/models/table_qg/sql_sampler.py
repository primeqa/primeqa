import math
from copy import deepcopy
import numpy as np
from primeqa.qg.utils.constants import SqlOperants, QGSpecialTokens

class SimpleSqlSampler():
    """ A simple sql sampler to sample sqls based on number of where clause conditions and other parameters
    """
    def __init__(self):
        self.sql_tokens = QGSpecialTokens

    @staticmethod
    def add_column_types(table):
        """Adds a data type list to the table dict based on values in the cells in that column.
        The data type for a column is either real or text.
        Args:
            table ([dict]): [The table Dict containing headers and rows]

        Returns:
            [Dict]: [Table Dict with  key 'type' containing list of data type for every column]
        """

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

    def sql_execution(self, where_clause, select_column, agg_op, table):
        """ This function executes the sql on a given table and returns the answer.

        Args:
            where_clause ([type]): [description]
            select_column ([type]): [description]
            agg_op ([type]): [description]
            table ([type]): [description]

        Returns:
            [String]: [Answer after executing sql on the given table]
        """
        

        selected_cells = []
        
        if (len(where_clause)>0):
            for row_id in where_clause['rows']:
                selected_cells.append(table['rows'][row_id][select_column])
        else:
            for row in table['rows']:
                selected_cells.append(row[select_column])
        
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
    
    @staticmethod
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

        # Many inequality conditions can be generated for a real column. This makes
        # it computationally expensive later when creating multiple where clauses. 
        # We will sample inequalities here for that reason.
        sampled_idx = np.random.choice(len(conds_list), min(
            num_conditions, len(conds_list)), replace=False)
        sampled_conds_list = [conds_list[i] for i in sampled_idx]

        return sampled_conds_list


    def _get_column_freq(self, table, if_ineq=False):
        """ Calculates frequency of a column in the table.

        Args:
            table ([Dict]): [Table Dictionary containing header and rows]
            if_ineq (bool, optional): [if there are inequality conditions or not]. Defaults to False.

        Returns:
            [List]: [Column List]
        """
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
                clist.extend(self._get_inequality_conds(col, num_ineq_conds))

            cols_list.append(clist)
        return cols_list

    
    def _check_condition(self, conds, cols_list):
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

    def _get_unique_conditions(self, wlist):
        wdict = {}
        for wc in wlist:
            conds_str = str(sorted([str(c) for c in wc['conds']]))
            wdict[conds_str] = wc
        wlist = []
        for key in wdict:
            wlist.append(wdict[key])
        return wlist

    def get_where_clauses(self, table, num_where=2, if_ineq=False):
        cols_list = self._get_column_freq(table, if_ineq)
        where_dict = {}
        if num_where==0:
            return where_dict

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
                            if self._check_condition(conds, cols_list):
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
                                if self._check_condition(conds, cols_list):
                                    intersection = list(
                                        set(w2['rows']) & set(cc[2]))
                                    wc = {'conds': conds, 'rows': intersection}
                                    where3_list.append(wc)
            where3_list = self._get_unique_conditions(where3_list)
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
                                if self._check_condition(conds, cols_list):
                                    intersection = list(
                                        set(w3['rows']) & set(cc[2]))
                                    wc = {'conds': conds, 'rows': intersection}
                                    where4_list.append(wc)
            where4_list = self._get_unique_conditions(where4_list)
            where_dict['nw-4'] = where4_list
        return where_dict


    def sample_sql(self, table, num_sample, num_where, agg_op=0, if_ineq=False):
        """ This function samples sqls from a given table based on values for the parameters
        num_sample -> number of sql queries to sample, num_where -> number of where condtioned desired in every sampled sql Query etc.
        Args:
            table ([Dict]): [Table dictionary with header and rows]
            num_sample ([int]): [Number of sqls to sample]
            num_where ([int]): [Number of where clause conditions every sampled sql should have]
            agg_op (int, optional): [Whether to sample aggregate queries or not]. Defaults to 0.
            if_ineq (bool, optional): [description]. Defaults to False.
            
        Returns:
            [List,Dict]: [Sampled sql query list in readable string format and dict format]
        """
        header = table['header']
        types = table['types']
        where_list = [] 
        multiple_where_dict = self.get_where_clauses(table, num_where, if_ineq)
        if num_where > 0 :
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

        if len(filtered_where_list) == 0 and num_where > 0 :
            return []
        
        sampled_where_list = []
        
        if num_where > 0:
            sample_ids = np.random.choice(
                len(filtered_where_list), num_sample, replace=True)
            sampled_where_list = [filtered_where_list[i] for i in sample_ids]

        if if_agg:
            clist = real_cols
        else:
            clist = list(range(len(header)))

        sql_string_list = []
        sql_list = []
        
        if (len(sampled_where_list)>0):
            for wc in sampled_where_list:
                try:
                    possible_cols = list(set(clist) - set([c[0] for c in wc['conds']]))
                    select_column = np.random.choice(possible_cols)
                    sql = {'sel': select_column, 'agg': agg_op}
                    sql['conds'] = wc['conds']
                    answer = self.sql_execution(wc, select_column, agg_op, table)
                    sql_dict = self.readable_sql(sql, header, answer)

                    sql_list.append(sql_dict)
                    sql_string_list.append(self.convert_sql_to_string(sql_dict, table))
                except:
                    print('\nQuery creation error')
        else :
            possible_cols = list(set(clist))
            select_column = np.random.choice(possible_cols)
            sql = {'sel': select_column, 'agg': agg_op}
            wc={}
            answer = self.sql_execution(wc, select_column, agg_op, table)
            sql_dict = self.readable_sql(sql, header, answer)
            sql_list.append(sql_dict)
            sql_string_list.append(self.convert_sql_to_string(sql_dict, table))


        return sql_string_list, sql_list

    def readable_sql(self, sql, header, answer):
        """Convert Non-readable SQL to readable SQL dict

        Args:
            sql ([Dict]): [Sql query]
            header ([List]): [Headers in table]
            answer ([String]): [Answer]

        Returns:
            [Dict]: [Sql Dict in readable format]
        """
        sql_dict = {}
        sql_dict['col'] = [SqlOperants.agg_ops[sql['agg']], header[sql['sel']]]
        conds = []
        if "conds" in sql :
            for c in sql['conds']:
                cond = [header[c[0]], SqlOperants.cond_ops_string[c[1]], str(c[2])]
                conds.append(cond)
            sql_dict['conds'] = conds
        sql_dict['answer'] = str(answer[0])
        return sql_dict

    def convert_sql_to_string(self, sql_dict, table=[], tokenizer='T5'):
        """Convert sql query in Dict format to string

        Args:
            sql_dict ([Dict]): [Sql Query to convert to string]
            table (list, optional): [Table]. Defaults to [].
            tokenizer (str, optional): [description]. Defaults to 'T5'.
        Returns:
            [String]: [Sql query in string format]
        """
        tokens = self.sql_tokens

        sql_str =  str(sql_dict['col'][0]) + ' '+tokens.sep+' ' + str(sql_dict['col'][1])
        if "conds" in sql_dict :
            for cond in sql_dict['conds']:
                sql_str += ' '+tokens.sep+' '
                sql_str += (' '+tokens.cond+' ').join([str(c) for c in cond])
        sql_str += ' '+tokens.ans+' ' + str(sql_dict['answer'])

        table['header'] = [str(h) for h in table['header']]
        sql_str += ' '+tokens.header+' ' + (' '+tokens.hsep+' ').join(table['header'])

        return sql_str

    def controlled_sample_sql(self, table_list, num_samples_per_table=5, agg_prob=[], num_where_prob=[], ineq_prob=0.0,id_list=[]):
        if agg_prob == []:
            # agg_prob = [0.6, 0.1, 0.1, 0.0, 0.1, 0.1] #['select', 'maximum', 'minimum', 'count', 'sum', 'average']
            # ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
            agg_prob = [1.0, 0., 0., 0., 0., 0.]
        if num_where_prob == []:
            # num_where_prob = [0.0, 0.45, 0.3, 0.2, 0.05] # [0,1,2,3,4], there is no zero where clause case
            # [0,1,2,3,4], there is no zero where clause case
            num_where_prob = [0.0, 1., 0., 0., 0.]

        sample_batch_size = 1
        sample_size_list = [sample_batch_size] * \
            math.floor(num_samples_per_table/sample_batch_size)
        if num_samples_per_table % sample_batch_size != 0:
            sample_size_list.append(num_samples_per_table % sample_batch_size)

        all_sql_str_list = []
        all_sql_list = []
        all_id_list = []
        # for table in tqdm(table_list):
        for i, table in enumerate(table_list):
            if 'types' not in table:
                table = self.add_column_types(table)
            for num_samples in sample_size_list:
                agg_op = np.random.choice(len(agg_prob), 1, True, agg_prob)[0]
                num_where = np.random.choice(
                    len(num_where_prob), 1, True, num_where_prob)[0]

                # if to use ineq.
                if_ineq = np.random.choice(2, 1, True, [1-ineq_prob, ineq_prob])[0]
                sql_str_list, sql_list = self.sample_sql(table, num_samples,
                                    num_where, agg_op, if_ineq)

                num_trials = 0
                while len(sql_str_list) < num_samples:
                    diff = num_samples - len(sql_str_list)
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
                        agg_op = np.random.choice([1, 2, 3, 4, 5])
                    diff_sql_str_list, diff_sql_list = self.sample_sql(
                                        table, diff, num_where, agg_op, if_ineq)
                    
                    sql_str_list.extend(diff_sql_str_list)
                    sql_list.extend(diff_sql_list)

                    num_trials += 1
                    if num_trials > 5:  # if we cant get it in 5 trials, let us skip this instances
                        print('Unsuccessful.')
                        break

                all_sql_str_list.extend(sql_str_list)
                if i < len(id_list) and id_list[i]!=None :
                    id_num = id_list[i]
                elif(len(id_list)>0):
                    id_num = "NA"
                if(len(id_list)>0):
                    id_num_list= [id_num] * len(sql_str_list)
                    all_id_list.extend(id_num_list)
                all_sql_list.extend(sql_list)
        return all_sql_str_list, all_sql_list, all_id_list