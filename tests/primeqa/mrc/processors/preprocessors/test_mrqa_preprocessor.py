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
        return ['SQuAD', 'TriviaQA', 'NaturalQuestionsSubset', 'NewsQA', 'SearchQA']

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

