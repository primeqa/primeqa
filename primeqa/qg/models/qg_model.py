import copy
import numpy as np

import torch
from torch import cuda

from primeqa.qg.utils.constants import QGSpecialTokens
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from primeqa.qg.models.hybrid_qg.path_sampler import PathSampler
from primeqa.qg.models.table_qg.sql_sampler import SimpleSqlSampler
from primeqa.qg.models.passage_qg.answer_sampler import AnswerSampler


class QGModel():
    def __init__(self,model_path, modality='table', lang='en'):
        """ Table Question Generation Model gets initialized based on either pre-trained model path or
        the model name. One example could be 't5-base'.

        Args:
            model_path (str): Either Name of the model or the path to the pre-trained model
            modality (str, optional): The modality specifies what data is predicted based on which input. Possible options include 'table' and 'passage'.
        """        
        if modality not in ['table', 'passage', 'hybrid']:
            raise NotImplementedError('This modality is not supported: ' + modality)

        self._device = torch.device('cuda') if cuda.is_available() else torch.device('cpu')
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        special_tokens_list = []

        self.modality = modality 
        if self.modality == 'passage':
            special_tokens_list.append(QGSpecialTokens.sep)
            self.answer_sampler = AnswerSampler()
        elif self.modality == 'table':
            special_tokens_list.extend([QGSpecialTokens.sep, QGSpecialTokens.cond, QGSpecialTokens.ans,
                            QGSpecialTokens.header, QGSpecialTokens.hsep])
            self.sql_sampler = SimpleSqlSampler()
        else:
            self.path_sampler = PathSampler(lang)

        # adding special tokens to tokenizer which will be used to convert SQL and Passage+Answer to string
        for special_token in special_tokens_list:
            if special_token not in self._tokenizer.vocab: # add only when special-tokens aren't already there
                self._tokenizer.add_tokens([special_token])
        # expanding token embeddings in model
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

    def prune_hallucinations(self, qdicts, num_instances=5, hallucination_prop=0.25):
        """
        Our approach to pruning hallucinated questions uses an entity lookup to check that 
        a generated entity or a part of it is present in the hybrid context. If the entity is not present, 
        the question and the entity are considered to be hallucinated.

        Args:
            qdicts (list): A list of dicts, each dict contains, an answer, context and (num_instances) generated questions.
        Returns:
            Updated qdicts without any hallucinated questions. Each dict will have only one question now.
        """
        new_qdicts = []
        reserve_hallucinated_qdict = []
        reserve_non_hallucinated_qdict = []
        question_words = set(['what', 'when', 'where', 'which', 'how', 'who', 'whose', 'whom'])
        for qdict in qdicts:
            answer = qdict['answer']
            context = qdict.pop('context')
            questions = qdict.pop('questions')
            if not reserve_hallucinated_qdict:
                qdict['question'] = questions[0]
                reserve_hallucinated_qdict.append(qdict)
            context_tokens = set([tok.text.lower() for tok in self.path_sampler.nlp_model.process(\
                                context, processors='tokenize').sentences[0].tokens])
            flag = False
            same_questions = set([])
            for question in questions:
                if not question.strip() or question in same_questions: continue
                same_questions.add(question)
                doc = self.path_sampler.nlp_model(question)
                hallucinated = set([])
                for entity in doc.sentences[0].entities:
                    entity = entity.text.lower()
                    entity_tokens = set([etok.text.strip().lower() for etok in self.path_sampler.nlp_model.process(\
                                        entity, processors='tokenize').sentences[0].tokens if etok.text.strip().lower() != "'s"])
                    has_qs_words = question_words.intersection(entity_tokens)
                    if has_qs_words: continue

                    hcount = 0
                    for e_tok in entity_tokens:
                        if e_tok not in context_tokens:
                            hcount += 1

                    if hcount == 0: continue

                    hall_quant = round(len(entity_tokens) * hallucination_prop) #no. of hall. words tolerable
                    if hcount > hall_quant or len(entity_tokens) == hcount:
                        hallucinated.add(entity)

                if not hallucinated:
                    qdict_copy = copy.deepcopy(qdict)
                    qdict_copy['question'] = question
                    if flag:
                        reserve_non_hallucinated_qdict.append(qdict_copy)
                    else:
                        new_qdicts.append(qdict_copy)
                    flag = True

        if not new_qdicts:
            #don't prune if all the generated questions are hallucinated 
            return reserve_hallucinated_qdict
        elif len(new_qdicts) < num_instances:
            #if number of questions are less than `num_instances` after pruning
            #use pruned questions to return number of questions equal to `num_instances`
            diff = num_instances - len(new_qdicts)
            np.random.shuffle(reserve_non_hallucinated_qdict) 
            return new_qdicts + reserve_non_hallucinated_qdict[:diff]
        else: 
            return new_qdicts

    def generate_questions(self, 
                data_list, 
                num_questions_per_instance=5, 
                agg_prob=[], 
                num_where_prob=[], 
                ineq_prob=0.0,
                hallucination_prop=0.25,
                num_beams=5,
                answers_list=[],
                id_list=[]):
                
        if type(data_list) == dict:
            data_list = [data_list]

        if self.modality == 'table':
            input_str_list, sql_list, id_question_list = self.sql_sampler.controlled_sample_sql(data_list, \
                        num_questions_per_instance, agg_prob, num_where_prob, ineq_prob, id_list)
            answer_list = [s['answer'] for s in sql_list]
        elif self.modality == 'passage':
            input_str_list, answer_list, id_question_list , id_context_map = self.answer_sampler.create_qg_input(data_list, \
                        num_questions_per_instance, answers_list, id_list)
        else:
            input_str_list, answer_list, id_question_list = self.path_sampler.create_qg_input(data_list, \
                        num_questions_per_instance, id_list)

        input_ids = self._tokenizer(input_str_list, 
                return_tensors='pt', 
                padding=True,
                truncation=True).to(self._device).input_ids
    
        if self.modality == "hybrid":
            num_return_sequences = num_beams
        else:
            num_return_sequences = 1

        generated_ids = self._model.generate(input_ids,
            max_length=60, 
            num_beams=num_beams,
            repetition_penalty=2.5,
            num_return_sequences=num_return_sequences,
            length_penalty=1.0,
            early_stopping=True)

        questions = [self._tokenizer.decode(g, skip_special_tokens=True,
                            clean_up_tokenization_spaces=True) for g in generated_ids]
        
        if id_question_list == [] :
            questions_dict = [{'question': questions[i], 'answer': answer_list[i]} for i in range(len(questions))]
        elif self.modality == 'passage' :
            questions_dict = [{'context_id':id_question_list[i], 'context':id_context_map.get(id_question_list[i]), \
                                'question': questions[i], 'answer': answer_list[i]} for i in range(len(questions))]
        else:
            assert len(questions) == len(input_str_list) * num_return_sequences
            questions_dict = [{'context_id':id_question_list[i], 'context': input_str_list[i], 'questions': list(qs),'answer': answer_list[i]} \
                                for i, qs in enumerate(np.split(np.array(questions), len(questions)//num_return_sequences))]
        if self.modality == 'hybrid':
            questions_dict = self.prune_hallucinations(questions_dict, hallucination_prop=hallucination_prop, \
                                num_instances=num_questions_per_instance)

        return questions_dict
