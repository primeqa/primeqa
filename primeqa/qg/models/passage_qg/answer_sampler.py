from primeqa.qg.utils.constants import QGSpecialTokens
import spacy
import numpy as np

class AnswerSampler():

    def __init__(self):
        self.nlp = spacy.load('xx_ent_wiki_sm')
        self.stopwords = self.nlp.Defaults.stop_words

    def _remove_stop_word_answers(self, text_list):
        return [t for t in text_list if t not in self.stopwords]
    
    def create_qg_input(self, data_list, num_questions_per_instance = 5):
        input_str_list = []
        answer_list = []
        for data in data_list:
            doc = self.nlp(data)
            answers = [n.text for n in doc.ents]
            if self.nlp.lang == 'en':
                noun_chunks = [n.text for n in doc.noun_chunks]
                answers = list(set(noun_chunks) | set(answers)) # taking union to remove repetition
            answers = self._remove_stop_word_answers(answers)
            
            if num_questions_per_instance < len(answers):
                answers = np.random.choice(answers, num_questions_per_instance, replace=False)
            
            for ans in answers:
                text = ans + ' '+QGSpecialTokens.sep+' ' + data        
                input_str_list.append(text)
                answer_list.append(ans)
        return input_str_list, answer_list
