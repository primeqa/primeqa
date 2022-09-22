import os
import tempfile

import pytest
from transformers import Seq2SeqTrainingArguments
from primeqa.mrc.data_models.data_collator import FiDDataCollator

from primeqa.mrc.trainers.seq2seq_mrc import MRCSeq2SeqTrainer
from tests.primeqa.mrc.common.base import UnitTest
from transformers import AutoConfig, AutoModel, AutoTokenizer
from primeqa.mrc.processors.preprocessors.eli5_fid import ELI5FiDPreprocessor
from primeqa.mrc.models.heads.generative import FID_HEAD
from primeqa.mrc.models.fid_task_model import FiDModelForDownstreamTasks
from datasets import Dataset


class TestMRCSeq2SeqTrainer(UnitTest):
    @pytest.fixture(scope='function')
    def training_args(self):
        with tempfile.TemporaryDirectory() as working_dir:
            yield Seq2SeqTrainingArguments(
                    output_dir=os.path.join(working_dir, 'output_dir'),
                    do_train=True,
                    do_eval=True,
                    num_train_epochs=1.5,
                    fp16=False,
                )
            
    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('facebook/bart-base')

    @pytest.fixture(scope='class')
    def preprocessor(self, tokenizer):
        return ELI5FiDPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            max_seq_len=256,
            max_contexts=3,
            max_answer_len=256
        )
            
    @pytest.fixture(scope='session')
    def eli5_train_examples(self):
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
        
    @pytest.fixture(scope='session')
    def eli5_eval_examples(self):
        return Dataset.from_dict(
            {
                "id": ["52py6m"], 
                "input": ["What causes the trail behind jets at high altitude?"], 
                "output": [
                    [
                    {"answer": "It is water vapor and ice. They are produced from the hot engine exhaust in the cold atmosphere. Water vapor from the engine exhaust mixed with unburnt particulate in the jet fuel gives the surrounding moist air something to latch onto and ice crystals form. Depending on the hight of the aircraft, they can last seconds to hours. If you have seen a running car on a brisk morning, that is a similar effect. The car is too close to the relatively warmer ground that trails do not last for more than a second"}, 
                    {"answer": "In moist air the compression and expansion of air around the wings can form a temporary cloud. However, what you see that lasts in the sky as a contrail is water vapor byproducts of the fuel combustion"}, 
                    ]
                ], 
                "passages": [
                    [
                        {
                            "pid": "387892::[4,4)",
                            "title": "Chemtrail conspiracy theory", 
                            "text": "Chemtrail conspiracy theory The chemtrail conspiracy theory is based on the erroneous belief that long-lasting condensation trails are \"chemtrails\" consisting of chemical or biological agents left in the sky by high-flying aircraft, sprayed for nefarious purposes undisclosed to the general public. Believers in this conspiracy theory say that while normal contrails dissipate relatively quickly, contrails that linger must contain additional substances. Those who subscribe to the theory speculate that the purpose of the chemical release may be solar radiation management, weather modification, psychological manipulation, human population control, or biological or chemical warfare, and that the trails are causing respiratory illnesse", 
                            "score": 82.02098846435547
                        }, 
                        {
                            "pid": "387892::[0,4)", 
                            "title": "Jet stream", 
                            "text": "Associated with jet streams is a phenomenon known as clear-air turbulence (CAT), caused by vertical and horizontal wind shear caused by jet streams. The CAT is strongest on the cold air side of the jet, next to and just under the axis of the jet. Clear-air turbulence can cause aircraft to plunge and so present a passenger safety hazard that has caused fatal accidents, such as the death of one passenger on United Airlines Flight 826. Section: Uses.:Possible future power generation.",
                            "score": 81.91938018798828
                        }, 
                        {
                            "pid": "1200025::[1,1)", 
                            "title": "Cirrus cloud", 
                            "text": "Contrails are a manmade type of cirrus cloud formed when water vapor from the exhaust of a jet engine condenses on particles, which come from either the surrounding air or the exhaust itself, and freezes, leaving behind a visible trail. The exhaust can also trigger the formation of cirrus by providing ice nuclei when there is an insufficient naturally-occurring supply in the atmosphere. One of the environmental impacts of aviation is that persistent contrails can form into large mats of cirrus, and increased air traffic has been implicated as one possible cause of the increasing frequency and amount of cirrus",
                            "score": 81.80261993408203
                        }
                    ]
                ]
            }
        )
        
    @pytest.fixture(scope='function')    
    def train_examples_and_features(self, eli5_train_examples, preprocessor):
        train_examples, train_features = preprocessor.process_train(eli5_train_examples)
        return train_examples, train_features
    
    @pytest.fixture(scope='function')    
    def eval_examples_and_features(self, eli5_eval_examples, preprocessor):
        train_examples, train_features = preprocessor.process_eval(eli5_eval_examples)
        return train_examples, train_features

    @pytest.fixture(scope='function')
    def data_collator(self, training_args, tokenizer):
        return FiDDataCollator(tokenizer, pad_to_multiple_of=64 if training_args.fp16 else None)
    
    @pytest.fixture(scope='function')
    def model_name_and_config(self):
        model_name = "facebook/bart-base"
        _ = AutoModel.from_pretrained(model_name)  # Pre-download LM inside flaky fixture so other tests have it
        return model_name, AutoConfig.from_pretrained(model_name)
    
    @pytest.fixture(scope='function')
    def config_and_model_with_generative_head(self, model_name_and_config):
        model_name, config = model_name_and_config
        model = FiDModelForDownstreamTasks.from_config(config,
                                                    model_name,
                                                    task_heads=FID_HEAD)
        head_name = next(iter(FID_HEAD))
        model.set_task_head(head_name)
        return config, model


    @pytest.fixture(scope='function')
    def trainer_with_generative_model(
            self, training_args, config_and_model_with_generative_head,
            tokenizer, train_examples_and_features, eval_examples_and_features, data_collator):
        _, model = config_and_model_with_generative_head
        _, train_features = train_examples_and_features
        eval_examples, eval_features = eval_examples_and_features
        return MRCSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_features if training_args.do_train else None,
            eval_dataset=eval_features if training_args.do_eval else None,
            eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=None,
            compute_metrics=None,
        )

    def test_trainer_trains_and_infers_without_errors(self, trainer_with_generative_model):
        train_result = trainer_with_generative_model.train()
        assert isinstance(train_result.training_loss, float)

        eval_metrics = trainer_with_generative_model.evaluate()
        assert eval_metrics == {}
