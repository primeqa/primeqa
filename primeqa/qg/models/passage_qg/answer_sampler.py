from primeqa.qg.utils.constants import QGSpecialTokens
import numpy as np
import stanza
from stanza.models.common.doc import Document
from stanza.pipeline.core import Pipeline


stanza.download(lang="multilingual")

class AnswerSampler():
    """
	Class for sampling answer tokens from passage, to be used for PassageQG inference.
	"""
    def __init__(self):
        """_summary_
        We use NERs from stanza library here. Currently we support four languages: Arabic, English, Finnish and Russian.
        These languages are in the intersection between TyDi QA data and Stanza NER.
        """
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
        for lang in self.lang_codes:
            if self.lang_codes[lang]['method_available'] == 'NER':
                self.ner_models[lang] = stanza.Pipeline(lang=lang, processors='tokenize,ner', verbose=False)
                print('Loaded NER model for ',self.lang_codes[lang]['name'])            
    
    def detect_language(self, text):
        """
        detect the language of an input passage, used later for identifying the right NER model to use
        """
        doc = Document([], text=text)
        self.lang_identify(doc)
        if doc.lang not in self.lang_codes:
            NotImplementedError('This language is not supported: ' + doc.lang)
        return doc.lang

    def get_named_entities(self, text):
        """
        pick the right NER model to use based on detected language of the passage
        """
        lang = self.detect_language(text)
        print('Input language', lang)
        if self.lang_codes[lang]['method_available'] == 'NER':
            output = self.ner_models[lang](text)
            return [ent.text for ent in output.ents]
        else:
            NotImplementedError
    
    def create_qg_input(self, 
                        data_list, 
                        num_questions_per_instance = 5, 
                        answers_list=[],
                        id_list=[]):
        """
        create the input for qg training: If the answers are provided in answers_list, use them. 
        Otherwise sampling named entities as possible answers from text passages.
        """
        input_str_list = []
        ans_list = []
        id_question_list = []
        id_context_map = dict()
        for i, data in enumerate(data_list):
            # If answers_list provided, then use them to generate questions. Otherwise use NER to
            # sample answers.
            if len(answers_list) > i and answers_list[i] != []: 
                answers = answers_list[i]
            else:
                answers = self.get_named_entities(data)
            if num_questions_per_instance < len(answers):
                answers = np.random.choice(answers, num_questions_per_instance, replace=False)
            for ans in answers:
                text = ans + ' '+QGSpecialTokens.sep+' ' + data        
                input_str_list.append(text)
                ans_list.append(ans)
                if len(id_list) > i and id_list[i] != None :
                    id_question_list.append(id_list[i])
                    id_context_map[id_list[i]]=data

        return input_str_list, ans_list, id_question_list, id_context_map
