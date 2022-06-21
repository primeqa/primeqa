from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from primeqa.qg.utils.constants import QGSpecialTokens
from primeqa.qg.models.table_qg.sql_sampler import SimpleSqlSampler


class QGModel():
    def __init__(self,model_path, modality='table'):
        """ Table Question Generation Model gets initialized based on either pre-trained model path or
        the model name. One example could be 't5-base'.

        Args:
            model_path (String): Either Name of the model or the path to the pre-trained model
        """        
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.modality = modality 

        # adding special tokens to tokenizer which will be used to convert SQL and Passage+Answer to string
        # expanding token embeddings in model
        
        sql_tokens_list = [QGSpecialTokens.sep, QGSpecialTokens.cond, QGSpecialTokens.ans,
                        QGSpecialTokens.header, QGSpecialTokens.hsep]
        for sql_token in sql_tokens_list:
            if sql_token not in self._tokenizer.vocab: # add only when special-tokens aren't already there
                self._tokenizer.add_tokens([sql_token])
        self._model.resize_token_embeddings(len(self._tokenizer.vocab))
    
    @property
    def model(self):
        """ Propery of TableQG model.

        Returns:
            Sequence to sequence model object (based on model name)
        """
        return self._model

    @property
    def tokenizer(self):
        """ Property of TableQG model.

        Returns:
            Tokenizer class object based on the model name/ path
        """
        return self._tokenizer

    def generate_questions(self, data_list, num_questions_per_instance=5, agg_prob=[], num_where_prob=[], ineq_prob=0.0):
        if type(data_list) == dict:
            data_list = [data_list]

        if self.modality == 'table':
            sql_sampler = SimpleSqlSampler()
            sql_string_list, sql_list = sql_sampler.controlled_sample_sql(data_list, num_questions_per_instance, agg_prob, num_where_prob, ineq_prob)

        input_ids = self._tokenizer(sql_string_list, 
            return_tensors='pt', 
            padding=True,
            truncation=True).input_ids

        generated_ids = self._model.generate(input_ids,
            max_length=60, # should go as an argument in this function and run_tableqg.py TODO
            num_beams=10,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True)

        questions = [self._tokenizer.decode(g, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True) for g in generated_ids]
        questions_dict = [{'question': questions[i], 'answer': sql_list[i]['answer']} for i in range(len(questions))]

        return questions_dict
