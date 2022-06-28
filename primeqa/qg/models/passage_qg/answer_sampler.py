from primeqa.qg.utils.constants import QGSpecialTokens
import spacy
import numpy as np

class AnswerSampler():

    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')
        self.stopwords = self.nlp.Defaults.stop_words

    def _remove_stop_word_answers(self, text_list):
        return [t for t in text_list if t not in self.stopwords]
    
    def create_qg_input(self, data_list, num_questions_per_instance = 5):
        input_str_list = []
        answer_list = []
        for data in data_list:
            doc = self.nlp(data)
            noun_chunks = [n.text for n in doc.noun_chunks]
            named_entities = [n.text for n in doc.ents]
            
            all_answers = list(set(noun_chunks) | set(named_entities)) # taking union to remove repetition
            all_answers = self._remove_stop_word_answers(all_answers)

            if num_questions_per_instance < len(all_answers):
                answers = np.random.choice(all_answers, num_questions_per_instance, replace=False)
            else:
                answers = all_answers
            
            for ans in answers:
                text = ans + ' '+QGSpecialTokens.sep+' ' + data        
                input_str_list.append(text)
                answer_list.append(ans)
        return input_str_list, answer_list
