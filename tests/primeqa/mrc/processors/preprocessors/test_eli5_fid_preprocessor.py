import datasets
from datasets import Dataset

import pytest
from transformers import AutoTokenizer
from primeqa.mrc.processors.preprocessors.eli5_fid import ELI5FiDPreprocessor
from tests.primeqa.mrc.common.base import UnitTest

class TestEli5Preprocessor(UnitTest):
    @pytest.fixture(scope='session')
    def eli5_examples(self):
        return Dataset.from_dict(
            {
                "id": ["1oy5tc"], 
                "input": ["in football whats the point of wasting the first two plays with a rush - up the middle - not regular rush plays i get those"], 
                "output": [
                    [
                    {"answer": "Keep the defense honest, get a feel for the pass rush, open up the passing game. An offense that's too one dimensional will fail. And those rushes up the middle can be busted wide open sometimes for big yardage.", "meta": {"score": 3}}, 
                    {"answer": "If you throw the ball all the time, then the defense will adapt to always cover for a pass.  By doing a simple running play every now and then, you force the defense to stay close and guard against the run.  Sometimes, the offense can catch the defense off guard by faking a run and freeing up their receivers.\n\nAlso, you don't have to gain massive yards on every single play.  Sometimes, it works best to gain a few yards at a time.  As long as you get the first down, you are in good shape.","meta": {"score": 3}}, 
                    {"answer": "In most cases the O-Line is supposed to make a hole for the running back to go through. If you run too many plays to the outside/throws the defense will catch on.\n\nAlso, 2 5 yard plays gets you a new set of downs.", "meta": {"score": 3}}, 
                    {"answer": "I you don't like those type of plays, watch CFL.  We only get 3 downs so you can't afford to waste one.  Lots more passing.", "meta": {"score": 2}}
                    ]
                ], 
                "passages": [
                    [
                        {
                            "pid": "387892::[4,4)",
                            "title": "Rush (gridiron football)", 
                            "text": "Rushing, on offense, is running with the ball when starting from behind the line of scrimmage with an intent of gaining yardage. While this usually means a running play, any offensive play that does not involve a forward pass is a rush - also called a run. It is usually done by the running back after a handoff from the quarterback, although quarterbacks and wide receivers can also rush. The quarterback will usually run when a passing play has broken down \u2013 such as when there is no receiver open to catch the ball \u2013 and there is room to", 
                            "score": 82.02098846435547
                        }, 
                        {
                            "pid": "387892::[0,4)", 
                            "title": "Rush (gridiron football)", 
                            "text": "Rush (gridiron football) Rushing is an action taken by the offense that means to advance the ball by running with it, as opposed to passing, or kicking. Any rushing player is called a rusher. Section: Running. Rushing, on offense, is running with the ball when starting from behind the line of scrimmage with an intent of gaining yardage. While this usually means a running play, any offensive play that does not involve a forward pass is a rush - also called a run. It is usually done by the running back after a handoff from the quarterback, although quarterbacks and", 
                            "score": 81.91938018798828
                        }, 
                        {
                            "pid": "1200025::[1,1)", 
                            "title": "Hurry-up offense", 
                            "text": "The hurry-up offense is an American football offensive style, which has two different but related forms in which the offensive team avoids delays between plays. The hurry-up, no-huddle offense (HUNH) refers to avoiding or shortening the huddle to limit or disrupt defensive strategies and flexibility. The two-minute drill is a clock-management strategy that may limit huddles but also emphasizes plays that stop the game clock. While the two-minute drill refers to parts of the game with little time remaining on the game clock, the no-huddle may be used in some form at any time. The no-huddle offense was pioneered by",
                            "score": 81.80261993408203
                        }
                    ]
                ]
            }
        )
        
    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('facebook/bart-base')

    @pytest.fixture(scope='class')
    def eli5_preprocessor(self, tokenizer):
        return ELI5FiDPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            max_seq_len=256,
            max_contexts=3,
            max_answer_len=256
        )
    
    def test_train_preprocessing_runs_without_errors(self, eli5_examples, eli5_preprocessor):
        train_examples, train_features = eli5_preprocessor.process_train(eli5_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        assert len(train_features['input_ids']) == 3
        assert len(train_features['labels']) == 3
        assert len(train_features['attention_mask']) == 3
        for example in train_features:
            assert len(example['input_ids']) == 3
    
    def test_eval_preprocessing_runs_without_errors(self, eli5_examples, eli5_preprocessor):
        eval_examples, eval_features = eli5_preprocessor.process_eval(eli5_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        assert len(eval_features['input_ids']) == 1
        assert len(eval_features['labels']) == 1
        assert len(eval_features['attention_mask']) == 1
        for example in eval_features:
            assert len(example['input_ids']) == 3