from operator import attrgetter

import pytest
from datasets import Dataset
from primeqa.al.models.al import ActiveLearner, GenALScorer, RCALScorer, TrainerConfig
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from primeqa.mrc.trainers.mrc import MRCTrainer
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader
from primeqa.qg.trainers.qg_trainer import GenTrainer
from primeqa.qg.utils.data_collator import DataCollatorForSeq2SeqWithDecoderInputs
from transformers import (
    AutoConfig,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    BartTokenizerFast,
    DataCollatorWithPadding,
    Seq2SeqTrainingArguments,
    TrainingArguments,
)


@pytest.mark.parametrize("model_name", ["facebook/bart-base"])
@pytest.mark.parametrize("score_function", ["sp", "dsp", "ls"])
def test_gen_scoring_functions(model_name, score_function):
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
            question=["To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"],
            answers=[{"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}],
        )
    )

    trainer_class = GenTrainer
    trainer_kwargs = dict(
        max_gen_length=30,
        model=model.model,
        tokenizer=model.tokenizer,
        args=Seq2SeqTrainingArguments(output_dir="tmp"),
        data_collator=DataCollatorForSeq2SeqWithDecoderInputs(model.tokenizer),
    )
    trainer = trainer_class(**trainer_kwargs)

    special_token_id_map = dict(
        bos_token_id=model.tokenizer.convert_tokens_to_ids("<q>"),
        eos_token_id=model.tokenizer.convert_tokens_to_ids("</q>"),
        bos_token_id_2=model.tokenizer.convert_tokens_to_ids("<a>"),
        eos_token_id_2=model.tokenizer.convert_tokens_to_ids("</a>"),
    )

    strategy = GenALScorer(GenALScorer.Strategy(score_function), 30, special_token_id_map, 0)
    strategy.init(data, {0: lambda data: (None, qgdl.create(data))})
    sample_ids = strategy.query(1, (trainer,), "id")
    assert len(sample_ids) == 1

    if score_function == "sp":
        # run AL test only once
        trainer_config = TrainerConfig(
            0,
            trainer_class,
            trainer_kwargs,
            lambda data: (None, qgdl.create(data)),
            lambda data: (None, qgdl.create(data)),
            None,
        )
        al = ActiveLearner("tmp", (trainer_config,))
        al.run(data, strategy, 1, 1, store_indices_and_scores=False, store_samples=False, feature_id_column="id")


@pytest.mark.parametrize("model_name", ["bert-base-uncased"])
def test_rc_scoring_function(model_name):
    data = Dataset.from_dict(
        dict(
            id=["0"],
            context=[
                'Architecturally, the school has a Catholic character. Atop the Main Building\'s gold dome is a golden statue of the Virgin Mary. Immediately in front of the Main Building and facing it, is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". Next to the Main Building is the Basilica of the Sacred Heart. Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. It is a replica of the grotto at Lourdes, France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous in 1858. At the end of the main drive (and in a direct line that connects through 3 statues and the Gold Dome), is a simple, modern stone statue of Mary.'
            ],
            question=["To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"],
            answers=[{"text": ["Saint Bernadette Soubirous"], "answer_start": [515]}],
        )
    )
    training_args = TrainingArguments(output_dir="tmp")

    task_heads = EXTRACTIVE_HEAD
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        config=config,
    )

    config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    model = ModelForDownstreamTasks.from_config(
        config,
        model_name,
        task_heads=task_heads,
    )
    model.set_task_head("qa_head")

    using_mixed_precision = any(attrgetter("fp16", "bf16")(training_args))
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

    # load preprocessor
    preprocessor_class = SQUADPreprocessor
    preprocessor = preprocessor_class(
        stride=128,
        tokenizer=tokenizer,
    )

    postprocessor_class = SQUADPostProcessor

    # noinspection PyProtectedMember
    postprocessor = postprocessor_class(
        k=1,
        n_best_size=20,
        max_answer_length=30,
    )

    trainer_class = MRCTrainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocessor.process_references_and_predictions,  # see QATrainer in Huggingface
    )
    trainer = trainer_class(**trainer_kwargs)

    strategy = RCALScorer(0)
    strategy.init(data, {0: preprocessor.process_eval})
    sample_ids = strategy.query(1, (trainer,), "id")
    assert len(sample_ids) == 1
