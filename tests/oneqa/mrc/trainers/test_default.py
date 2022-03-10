import os
import tempfile

import pytest
from transformers import TrainingArguments, DataCollatorWithPadding

from oneqa.mrc.trainers.default import MRCTrainer
from tests.oneqa.mrc.common.base import UnitTest


class TestMRCTrainer(UnitTest):
    @pytest.fixture(scope='function')
    def training_args(self):
        with tempfile.TemporaryDirectory() as working_dir:
            yield TrainingArguments(
                    output_dir=os.path.join(working_dir, 'output_dir'),
                    do_train=True,
                    do_eval=False,  # TODO: trainer test eval
                    num_train_epochs=1,
                    fp16=False,
                )

    @pytest.fixture()
    def train_dataset(self):
        raise NotImplementedError

    @pytest.fixture()
    def eval_dataset(self):
        return None  # TODO: trainer test eval dataset

    @pytest.fixture()
    def data_collator(self, training_args, tokenizer):
        return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if training_args.fp16 else None)

    @pytest.fixture()
    def trainer_with_extractive_model(
            self, training_args, config_and_model_with_extractive_head,
            tokenizer, train_dataset, eval_dataset, data_collator):
        _, model = config_and_model_with_extractive_head
        return MRCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            # eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # post_process_function=postprocessor.process,
            compute_metrics=None,
        )

    @pytest.mark.skip()
    def test_trainer(self, trainer_with_extractive_model):
        # raise ValueError(f"{trainer_with_extractive_model}")
        raise NotImplementedError
