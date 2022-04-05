import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

from tableQG.sql_sampler import controlled_sample_sql
from tableQG.utils import add_column_types
from tableQG.t5_generation import inference

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'


class GenerateQuestions:
    def __init__(self, model_path):
        self.device = 'cuda' if cuda.is_available() else 'cpu'

        self.tokenizer = T5Tokenizer.from_pretrained("t5-base")
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        print("loaded")

    def _sample_sql(self, table_list, num_samples_per_table=5, agg_prob=[], num_where_prob=[], ineq_prob=0.0):
        # agg_prob: a probability vector of length 6, with each index giving prob of an
        # aggregate ops appearing in this order ['select', 'maximum', 'minimum', 'count', 'sum', 'average']

        # num_where_prob: a vector of size 5 with probablities of number of where clauses to use while generating
        # sqls. If k where clause can't be generated the code tries to generate k-1 where clause query, and so on.

        # if_ineq

        # default is SELECT query with ONE where clause
        if agg_prob == []:
            # ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
            agg_prob = [1.0, 0., 0., 0., 0., 0.]
        if num_where_prob == []:
            # [0,1,2,3,4], there is no zero where clause case
            num_where_prob = [0.0, 1., 0., 0., 0.]
       # print("table_list: next "+table_list)
        # adding column type if missing
        if 'type' not in table_list[0]:
            table_list = [add_column_types(t) for t in table_list]

        # adding table ids
        for i, t in enumerate(table_list):
            t['id'] = i

        return controlled_sample_sql(table_list, num_samples_per_table, agg_prob, num_where_prob, ineq_prob)

    def generate_question(self, table_list, num_samples_per_table=5, agg_prob=[], num_where_prob=[], ineq_prob=0.0):
        use_col = True  # if to use column-headers while generating questions.

        sql_list = self._sample_sql(
            table_list, num_samples_per_table, agg_prob, num_where_prob, ineq_prob)

        question_list = []
        for sql_dict in sql_list:
            qdict = {}
            qdict['sql'] = sql_dict
            qdict['answer'] = sql_dict['answer']
            questions = inference(sql_dict, self.model, self.tokenizer,
                                  table_list[sql_dict['table_id']], use_col=True)
            qdict['question'] = questions[0]
            question_list.append(qdict)

        return question_list
