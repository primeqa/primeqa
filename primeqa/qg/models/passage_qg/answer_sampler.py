from primeqa.qg.utils.constants import QGSpecialTokens
import numpy as np
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline


stanza.download(lang="multilingual")

class AnswerSampler():
    def __init__(self):
        self.lang_codes = {'ar': {'name':'Arabic', 'method_available': 'NER'},
                           'en': {'name':'English', 'method_available':'NER'},
                           'fi': {'name':'Finnish', 'method_available':'NER'},
                           'id': {'name':'Indonesian', 'method_available':'UD'},
                           'ko': {'name':'Korean', 'method_available':'UD'},
                           'ru': {'name':'Russian', 'method_available':'NER'},
                           'te': {'name':'Telugu', 'method_available':'UD'}
                        }
        self.lang_identify = Pipeline(lang="multilingual", processors="langid", verbose=False)
        self.ner_models = {}
        for code in self.lang_codes:
            if self.lang_codes[code]['method_available'] == 'NER':
                self.ner_models[code] = stanza.Pipeline(lang=code, processors='tokenize,ner', verbose=False)
                print('Loaded NER model for ',self.lang_codes[code]['name'])

    def detect_language(self, text):
        doc = Document([], text=text)
        self.lang_identify(doc)
        if doc.lang not in self.lang_codes:
            NotImplementedError('This language is not supported: ' + doc.lang)
        return doc.lang

    def get_named_entities(self, text):
        lang = self.detect_language(text)
        print('Input language', lang)
        if self.lang_codes[lang]['method_available'] == 'NER':
            output = self.ner_models[lang](text)
            return [ent.text for ent in output.ents]
        else:
            NotImplementedError
    
    def create_qg_input(self, data_list, num_questions_per_instance = 5):
        input_str_list = []
        answer_list = []
        for data in data_list:
            answers = self.get_named_entities(data)

            if num_questions_per_instance < len(answers):
                answers = np.random.choice(answers, num_questions_per_instance, replace=False)
            
            for ans in answers:
                text = ans + ' '+QGSpecialTokens.sep+' ' + data        
                input_str_list.append(text)
                answer_list.append(ans)
        return input_str_list, answer_list
