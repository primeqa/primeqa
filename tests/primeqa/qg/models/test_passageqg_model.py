import pytest
from primeqa.qg.models.qg_model import QGModel
from transformers import T5Tokenizer, T5ForConditionalGeneration


@pytest.mark.parametrize("model_name",["t5-base"])
def test_qg_model(model_name):
    tqm = QGModel(model_name, modality='passage')
    assert type(tqm.model)==T5ForConditionalGeneration
    assert type((tqm.tokenizer)==T5Tokenizer)

    text_list = ["Sachin tendulkar was an Indian cricketer born in Mumbai. He scored nearly 350000 runs in his international career"]

    id_list = [ "xyzID456"]

    question_dict = tqm.generate_questions(text_list, 
                    num_questions_per_instance = 2, id_list=id_list)

    assert(len(question_dict) > 0)
