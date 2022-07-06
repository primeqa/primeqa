import pytest
from primeqa.qg.models.qg_model import QGModel
from transformers import T5Tokenizer, T5ForConditionalGeneration

@pytest.mark.parametrize("model_name",["t5-small"])
def test_qg_model(model_name):
    tqm = QGModel(model_name, modality='table')
    assert type(tqm.model)==T5ForConditionalGeneration
    assert type((tqm.tokenizer)==T5Tokenizer)

    table_list = [
        {
            "header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School Team"],
            "rows": [
                ["Antonio Lang", 21, "United States", "Guard-Forward", "1999-2000", "Duke"],
                ["Voshon Lenard", 2, "United States", "Guard", "2002-03", "Minnesota"],
                ["Martin Lewis", 32, "United States", "Guard-Forward", "1996-97", "Butler CC (KS)"],
                ["Brad Lohaus", 33, "United States", "Forward-Center", "1996", "Iowa"],
                ["Art Long", 42, "United States", "Forward-Center", "2002-03", "Cincinnati"]
            ]
        }
    ]
    gqs= tqm.generate_questions(table_list, 
                            num_questions_per_instance = 10,
                            agg_prob = [1.,0,0,0,0,0],
                            num_where_prob = [0,1.,0,0,0],
                            ineq_prob = 0.0)
    assert(len(gqs)>0)
    

