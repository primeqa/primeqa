from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from primeqa.qg.utils.constants import QGSpecialTokens
from primeqa.qg.models.table_qg.sql_sampler import SimpleSqlSampler
from primeqa.qg.models.passage_qg.answer_sampler import AnswerSampler



class QGModel():
    def __init__(self,model_path, modality='table'):
        """ Table Question Generation Model gets initialized based on either pre-trained model path or
        the model name. One example could be 't5-base'.

        Args:
            model_path (String): Either Name of the model or the path to the pre-trained model
        """        
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        # adding special tokens to tokenizer which will be used to convert SQL and Passage+Answer to string
        # expanding token embeddings in model
        
        sql_tokens_list = [QGSpecialTokens.sep, QGSpecialTokens.cond, QGSpecialTokens.ans,
                        QGSpecialTokens.header, QGSpecialTokens.hsep]
        for sql_token in sql_tokens_list:
            if sql_token not in self._tokenizer.vocab: # add only when special-tokens aren't already there
                self._tokenizer.add_tokens([sql_token])
        self._model.resize_token_embeddings(len(self._tokenizer.vocab))

        self.modality = modality 
        if self.modality == 'passage':
            self.answer_sampler = AnswerSampler()
        elif self.modality == 'table':
            self.sql_sampler = SimpleSqlSampler()

    
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

    def generate_questions(self, 
                data_list, 
                num_questions_per_instance=5, 
                agg_prob=[], 
                num_where_prob=[], 
                ineq_prob=0.0,
                answers_list=[],
                id_list=[]):
                
        if type(data_list) == dict:
            data_list = [data_list]

        if self.modality == 'table':
            input_str_list, sql_list, id_question_list = self.sql_sampler.controlled_sample_sql(data_list, num_questions_per_instance, agg_prob, num_where_prob, ineq_prob, id_list)
            answer_list = [s['answer'] for s in sql_list]
        elif self.modality == 'passage':
            input_str_list, answer_list, id_question_list , id_context_map = self.answer_sampler.create_qg_input(data_list, num_questions_per_instance, answers_list, id_list)

        input_ids = self._tokenizer(input_str_list, 
            return_tensors='pt', 
            padding=True,
            truncation=True).input_ids

        generated_ids = self._model.generate(input_ids,
            max_length=60, 
            num_beams=10,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True)

        questions = [self._tokenizer.decode(g, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True) for g in generated_ids]
        
        if id_question_list == [] :
            questions_dict = [{'question': questions[i], 'answer': answer_list[i]} for i in range(len(questions))]
        elif self.modality == 'passage' :
            questions_dict = [{'context_id':id_question_list[i], 'context':id_context_map.get(id_question_list[i]),'question': questions[i], 'answer': answer_list[i]} for i in range(len(questions))]
        else:
            questions_dict = [{'context_id':id_question_list[i], 'question': questions[i], 'answer': answer_list[i]} for i in range(len(questions))]
        
        return questions_dict
