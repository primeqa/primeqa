from oneqa.tableqg.models.tableqg_model import TableQG
from oneqa.tableqg.models.sql_sampler import SimpleSqlSampler

class QGGeneration:
	def __init__(self, model_name_or_path):
		self.tqg = TableQG(model_name_or_path)

	def generate_questions(self, table_list, num_samples_per_table=5, agg_prob=[], num_where_prob=[], ineq_prob=0.0):
		if type(table_list) == dict:
			table_list = [table_list]

		sql_sampler = SimpleSqlSampler()
		sql_string_list, sql_list = sql_sampler.controlled_sample_sql(table_list, num_samples_per_table, agg_prob, num_where_prob, ineq_prob)
		
		input_ids = self.tqg.tokenizer(sql_string_list, 
			return_tensors='pt', 
			padding=True,
			truncation=True).input_ids

		generated_ids = self.tqg.model.generate(input_ids,
		        max_length=60, # should go as an argument in this function and run_tableqg.py TODO
				num_beams=10,
				repetition_penalty=2.5,
				length_penalty=1.0,
				early_stopping=True)
		
		questions = [self.tqg.tokenizer.decode(g, skip_special_tokens=True,
                              clean_up_tokenization_spaces=True) for g in generated_ids]
		questions_dict = [{'question': questions[i], 'answer': sql_list[i]['answer']} for i in range(len(questions))]
		
		return questions_dict