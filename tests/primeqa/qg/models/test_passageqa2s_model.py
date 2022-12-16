import pytest
from datasets import Dataset
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader
from primeqa.qg.trainers.qg_trainer import GenTrainer
from primeqa.qg.utils.data_collator import DataCollatorForSeq2SeqWithDecoderInputs
from transformers import BartForConditionalGeneration, BartTokenizer, BartTokenizerFast, Seq2SeqTrainingArguments


@pytest.mark.parametrize("model_name", ["facebook/bart-base"])
def test_qa2s_model(model_name):
    model = QGModel(model_name, modality="passage", gen_config="qa2s")
    assert isinstance(model.model, BartForConditionalGeneration)
    assert isinstance(model.tokenizer, (BartTokenizer, BartTokenizerFast))

    qgdl = QGDataLoader(
        tokenizer=model.tokenizer,
        modality="passage",
        gen_config="qa2s",
        input_max_len=None,
        target_max_len=None,
    )
    data = Dataset.from_dict(
        dict(
            id=[0],
            context=[
                'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
            ],
        )
    )
    data = qgdl.create(data)

    trainer = GenTrainer(
        max_gen_length=30,
        model=model.model,
        tokenizer=model.tokenizer,
        args=Seq2SeqTrainingArguments(output_dir="tmp"),
        data_collator=DataCollatorForSeq2SeqWithDecoderInputs(model.tokenizer),
    )

    predictions = trainer.predict(test_dataset=data)

    assert len(predictions) > 0
