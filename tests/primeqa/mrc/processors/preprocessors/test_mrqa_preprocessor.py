import datasets
from datasets import Dataset
from datasets.features.features import Sequence, Value, ClassLabel
import pytest
from pytest import raises
from transformers import AutoTokenizer
from primeqa.mrc.processors.preprocessors.mrqa import MRQAPreprocessor
from tests.primeqa.mrc.common.base import UnitTest


class TestMRQAPreprocessor(UnitTest):
    @pytest.fixture(scope='session')
    def mrqa_subsets(self):
        return ['SQuAD', 'TriviaQA', 'NaturalQuestionsSubset', 'NewsQA', 'SearchQA', 'HotpotQA']

    @pytest.fixture(scope='session')
    def mrqa_examples(self):
        features=datasets.Features(
            {
            'subset': Value(dtype='string', id=None), 
            'context': Value(dtype='string', id=None), 
            'context_tokens': Sequence(feature={'tokens': Value(dtype='string', id=None), 'offsets': Value(dtype='int32', id=None)}, length=-1, id=None), 
            'qid': Value(dtype='string', id=None), 
            'question': Value(dtype='string', id=None), 
            'question_tokens': Sequence(feature={'tokens': Value(dtype='string', id=None), 'offsets': Value(dtype='int32', id=None)}, length=-1, id=None), 
            'detected_answers': Sequence(feature={'text': Value(dtype='string', id=None), 
            'char_spans': Sequence(feature={'start': Value(dtype='int32', id=None), 'end': Value(dtype='int32', id=None)}, length=-1, id=None), 
            'token_spans': Sequence(feature={'start': Value(dtype='int32', id=None), 'end': Value(dtype='int32', id=None)}, length=-1, id=None)}, length=-1, id=None), 
            'answers': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None)
            }
        )
        hotpotqa_example = {'subset': 'HotpotQA', 
                            'context': '[PAR] [TLE] The Oberoi Group [SEP] The Oberoi Group is a hotel company with its head office in Delhi.  Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands. [PAR] [TLE] Oberoi family [SEP] The Oberoi family is an Indian family that is famous for its involvement in hotels, namely through The Oberoi Group.', 
                            'context_tokens': {'tokens': ['[PAR]', '[TLE]', 'The', 'Oberoi', 'Group', '[', 'SEP', ']', 'The', 'Oberoi', 'Group', 'is', 'a', 'hotel', 'company', 'with', 'its', 'head', 'office', 'in', 'Delhi', '.', 'Founded', 'in', '1934', ',', 'the', 'company', 'owns', 'and/or', 'operates', '30', '+', 'luxury', 'hotels', 'and', 'two', 'river', 'cruise', 'ships', 'in', 'six', 'countries', ',', 'primarily', 'under', 'its', 'Oberoi', 'Hotels', '&', 'Resorts', 'and', 'Trident', 'Hotels', 'brands', '.', '[PAR]', '[TLE]', 'Oberoi', 'family', '[', 'SEP', ']', 'The', 'Oberoi', 'family', 'is', 'an', 'Indian', 'family', 'that', 'is', 'famous', 'for', 'its', 'involvement', 'in', 'hotels', ',', 'namely', 'through', 'The', 'Oberoi', 'Group', '.'], 'offsets': [0, 6, 12, 16, 23, 29, 30, 33, 35, 39, 46, 52, 55, 57, 63, 71, 76, 80, 85, 92, 95, 100, 103, 111, 114, 118, 120, 124, 132, 137, 144, 153, 155, 157, 164, 171, 175, 179, 185, 192, 198, 201, 205, 214, 216, 226, 232, 236, 243, 250, 252, 260, 264, 272, 279, 285, 287, 293, 299, 306, 313, 314, 317, 319, 323, 330, 337, 340, 343, 350, 357, 362, 365, 372, 376, 380, 392, 395, 401, 403, 410, 418, 422, 429, 434]}, 
                            'qid': '197390227efc4f86b313a2b585cd685b', 
                            'question': 'The Oberoi family is part of a hotel company that has a head office in what city?', 
                            'question_tokens': {'tokens': ['The', 'Oberoi', 'family', 'is', 'part', 'of', 'a', 'hotel', 'company', 'that', 'has', 'a', 'head', 'office', 'in', 'what', 'city', '?'], 'offsets': [0, 4, 11, 18, 21, 26, 29, 31, 37, 45, 50, 54, 56, 61, 68, 71, 76, 80]}, 
                            'detected_answers': {'text': ['Delhi'], 'char_spans': [{'start': [95], 'end': [99]}], 'token_spans': [{'start': [20], 'end': [20]}]}, 
                            'answers': ['Delhi']}
        
        
        qids = ['squad-ex', 'triviaqa-ex', 'nq-ex', 'newsqa-ex', 'searchqa-ex']
        subsets = ['SQuAD', 'TriviaQA', 'NaturalQuestionsSubset', 'NewsQA', 'SearchQA']
        questions = ["Who was the main performer at this year's halftime show?"] * 5
        question_tokens = {
                    'offsets': [0, 4, 8, 12, 17, 27, 30, 35, 39, 42, 51, 55],
                    'tokens': ['Who', 'was', 'the', 'main', 'performer', 'at', 'this', 'year', "'s", 'halftime', 'show', '?']
                }
        qtokens_list = [question_tokens] * 5
        
        
        contexts = ['CBS broadcast Super Bowl 50 in the U.S., and charged an average of $5 million for a 30-second commercial during the game. The Super Bowl 50 halftime show was headlined by the British rock group Coldplay with special guest performers Beyoncé and Bruno Mars, who headlined the Super Bowl XLVII and Super Bowl XLVIII halftime shows, respectively. It was the third-most watched U.S. broadcast ever.'] * 5
        context_tokens = {
                    'offsets': [0, 4, 14, 20, 25, 28, 31, 35, 39, 41, 45, 53, 56, 64, 67, 68, 70, 78, 82, 84, 94, 105, 112, 116, 120, 122, 126, 132, 137, 140, 149, 154, 158, 168, 171, 175, 183, 188, 194, 203, 208, 216, 222, 233, 241, 245, 251, 255, 257, 261, 271, 275, 281, 286, 292, 296, 302, 307, 314, 323, 328, 330, 342, 344, 347, 351, 355, 360, 361, 366, 374, 379, 389, 393],
                    'tokens': ['CBS', 'broadcast', 'Super', 'Bowl', '50', 'in', 'the', 'U.S.', ',', 'and', 'charged', 'an', 'average', 'of', '$', '5', 'million', 'for', 'a', '30-second', 'commercial', 'during', 'the', 'game', '.', 'The', 'Super', 'Bowl', '50', 'halftime', 'show', 'was', 'headlined', 'by', 'the', 'British', 'rock', 'group', 'Coldplay', 'with', 'special', 'guest', 'performers', 'Beyoncé', 'and', 'Bruno', 'Mars', ',', 'who', 'headlined', 'the', 'Super', 'Bowl', 'XLVII', 'and', 'Super', 'Bowl', 'XLVIII', 'halftime', 'shows', ',', 'respectively', '.', 'It', 'was', 'the', 'third', '-', 'most', 'watched', 'U.S.', 'broadcast', 'ever', '.']
                }
        ctokens_list = [context_tokens] * 5

        detected_answers = {
                    'char_spans': [
                    {
                        'end': [201],
                        'start': [194]
                    }, {
                        'end': [201],
                        'start': [194]
                    }, {
                        'end': [201],
                        'start': [194]
                    }
                    ],
                    'text': ['Coldplay', 'Coldplay', 'Coldplay'],
                    'token_spans': [
                    {
                        'end': [38],
                        'start': [38]
                    }, {
                        'end': [38],
                        'start': [38]
                        }, {
                        'end': [38],
                        'start': [38]
                    }
                    ]
                }
        detected_answers_list = [detected_answers] * 5
        answers = ['Coldplay', 'Coldplay', 'Coldplay']
        answer_list = [answers] * 5
        
        # add HotpotQA example
        qids.append('hotpotqa-ex')
        subsets.append(hotpotqa_example['subset'])
        questions.append(hotpotqa_example['question'])
        qtokens_list.append(hotpotqa_example['question_tokens'])
        contexts.append(hotpotqa_example['context'])
        ctokens_list.append(hotpotqa_example['context_tokens'])
        detected_answers_list.append(hotpotqa_example['detected_answers'])
        answer_list.append(hotpotqa_example['answers'])
        
        
        
        examples_dict = dict(qid=qids, subset=subsets, question=questions, context=contexts, 
                                question_tokens=qtokens_list,context_tokens=ctokens_list, 
                                detected_answers=detected_answers_list,answers=answer_list)
        examples = datasets.Dataset.from_dict(examples_dict)
        return examples
    

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('roberta-base')
    
    @pytest.fixture(scope='class')
    def mrqa_preprocessor(self, tokenizer):
        return MRQAPreprocessor(
                tokenizer,
                stride=128,
                load_from_cache_file=False,
            )

    def test_train_preprocessing_runs_without_errors(self, mrqa_examples, mrqa_preprocessor):
        train_examples, train_features = mrqa_preprocessor.process_train(mrqa_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        for example in train_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1

    def test_filter_subset(self, mrqa_examples, mrqa_subsets):
        for subset in mrqa_subsets:
            mrqa_subset_examples =  mrqa_examples.filter(lambda example: example['subset'] in [subset])
            assert mrqa_subset_examples.num_rows == 1
            assert mrqa_subset_examples[0]['subset'] == subset
    
    def test_filter_multiple_subsets(self, mrqa_examples, mrqa_subsets):
        subsets = [mrqa_subsets[0], mrqa_subsets[-1]]
        mrqa_subset_examples =  mrqa_examples.filter(lambda example: example['subset'] in subsets)
        assert mrqa_subset_examples.num_rows == 2
        assert mrqa_subset_examples[0]['subset'] == subsets[0]
        assert mrqa_subset_examples[1]['subset'] == subsets[1]

