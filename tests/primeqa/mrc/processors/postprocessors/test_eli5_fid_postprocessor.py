import datasets
from datasets import Dataset
import pytest
from transformers import AutoTokenizer
from transformers.trainer_utils import EvalLoopOutput
import numpy as np
from primeqa.mrc.processors.postprocessors.eli5_fid import ELI5FiDPostProcessor
from primeqa.mrc.processors.preprocessors.eli5_fid import ELI5FiDPreprocessor
from tests.primeqa.mrc.common.base import UnitTest

class TestELI5FiDPostProcessor(UnitTest):
    
    @pytest.fixture(scope='session')
    def eli5_examples(self):
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
        
    _expected_predictions = "The water vapor from the exhaust of the jet is condensing on the surrounding air, which comes from either the surrounding water vapor or the exhaust itself, leaving behind a visible trail. \n\nThe contrails are a manmade type of cirrus cloud formed when water vapor and ice nuclei are frozen and form a large ice sheet."
    
    @pytest.fixture(scope='session')
    def mrc_output(self):
        predictions = np.array([[2, 0, 0, 0, 133, 514, 29406, 31, 5, 19379, 9, 5, 4900, 16, 10022, 21591, 15, 5, 3817, 935, 6, 61, 606, 31, 1169, 5, 3817, 514, 29406, 50, 5, 19379, 1495, 6, 1618, 639, 10, 7097, 5592, 4, 1437, 50118, 50118, 133, 15441, 5290, 32, 10, 313, 7078, 1907, 9, 40441, 14888, 3613, 4829, 77, 514, 29406, 8, 2480, 38898, 118, 32, 9214, 8, 1026, 10, 739, 2480, 5462, 4, 2, 1, 1 ]])
        return EvalLoopOutput(predictions=predictions, label_ids=None, metrics=None, num_samples=1)
    
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
        
    @pytest.fixture(scope='class')
    def eli5_eval_examples_and_features(self, eli5_examples, eli5_preprocessor):
        return eli5_preprocessor.process_eval(eli5_examples)


    def test_post_processor_has_examples_and_features(self, eli5_eval_examples_and_features,tokenizer,
                                                      mrc_output):
        postprocessor_class = ELI5FiDPostProcessor(k=1, tokenizer=tokenizer, max_answer_length=126,
                                                            single_context_multiple_passages=True)
        eval_examples, eval_features = eli5_eval_examples_and_features
        eli5_predictions = postprocessor_class.process(eval_examples, eval_features, mrc_output)
        assert eli5_predictions[0]['prediction_text'] == self._expected_predictions