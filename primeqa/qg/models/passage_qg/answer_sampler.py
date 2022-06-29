from primeqa.qg.utils.constants import QGSpecialTokens
import spacy
import numpy as np

class AnswerSampler():
    """
	Class for sampling answer tokens from passage, to be used for PassageQG
	"""

    def __init__(self):
        """
        load a multi-lingual nlp model to be used for NER taggging and stopword detection
        """

        self.nlp = spacy.load('xx_ent_wiki_sm')
        self.stopwords = self.nlp.Defaults.stop_words

    def _remove_stop_word_answers(self, text_list):
        return [t for t in text_list if t not in self.stopwords]
    
    def create_qg_input(self, data_list, num_questions_per_instance = 5):
        """
        create the input for qg training by sampling noun phrases as possible
        answers from text passages.
        """

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
                text = ans + ' '+QGSpecialTokens.sep+' ' + data    # prepare the data to match qg training format  
                input_str_list.append(text)
                answer_list.append(ans)
        return input_str_list, answer_list
