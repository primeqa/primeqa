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
                    num_train_epochs=1.5,
                    fp16=False,
                )

    @pytest.fixture()
    def data_collator(self, training_args, tokenizer):
        return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if training_args.fp16 else None)

    @pytest.fixture()
    def trainer_with_extractive_model(
            self, training_args, config_and_model_with_extractive_head,
            tokenizer, train_examples_and_features, eval_examples_and_features, data_collator):
        _, model = config_and_model_with_extractive_head
        _, train_features = train_examples_and_features
        _, eval_features = eval_examples_and_features
        return MRCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_features if training_args.do_train else None,
            eval_dataset=eval_features if training_args.do_eval else None,
            # eval_examples=eval_examples if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # post_process_function=postprocessor.process,
            compute_metrics=None,
        )

    def test_trainer_trains_without_errors(self, trainer_with_extractive_model):
        train_result = trainer_with_extractive_model.train()
        assert isinstance(train_result.training_loss, float)
