import os
import tempfile

from datasets import Dataset
from torch.utils.data import ConcatDataset

import pytest
from transformers import TrainingArguments, DataCollatorWithPadding

from primeqa.mrc.trainers.mrc_mskd import MSKD_MRCTrainer
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from tests.primeqa.mrc.common.base import UnitTest


class TestMSKD_MRCTrainer(UnitTest):
    @pytest.fixture(scope='function')
    def training_args(self):
        with tempfile.TemporaryDirectory() as working_dir:
            yield TrainingArguments(
                    output_dir=os.path.join(working_dir, 'output_dir'),
                    do_train=True,
                    do_eval=True,
                    num_train_epochs=1.5,
                    fp16=False,
                )

    @pytest.fixture(scope='function')
    def data_collator(self, training_args, tokenizer):
        return DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if training_args.fp16 else None)

    @pytest.fixture(scope='session')
    def train_examples(self):
        question = ["Who walked the dog?", "What time is it?"]
        context = [["Alice walks the cat", "Bob walks the dog"],
                   ["The quick brown fox jumps over the lazy dog", "Glenn the otter lives at the aquarium", "Go"]]
        example_id = ["foo-abc", "bar-123"]
        start_positions = [[0], [-1]]
        end_positions = [[2], [-1]]
        passage_indices = [[1], [-1]]
        yes_no_answer = [["NONE"], ["NONE"]]
        examples_dict = dict(question=question, context=context, example_id=example_id,
                             target=[dict(start_positions=s, end_positions=e, passage_indices=p, yes_no_answer=yn)
                                     for s, e, p, yn in
                                     zip(start_positions, end_positions, passage_indices, yes_no_answer)])
        examples_dataset1 = Dataset.from_dict(examples_dict)
        examples_dataset2 = Dataset.from_dict(examples_dict)
        examples_dataset3 = Dataset.from_dict(examples_dict)
        examples_datasets = [examples_dataset1, examples_dataset2, examples_dataset3]
        return examples_datasets

    @pytest.fixture(scope='session')
    def eval_examples(self, train_examples):
        ees = []
        for te in train_examples:
            ee = te.remove_columns('target')
            ees.append(ee)
        return ees

    @pytest.fixture(scope='session')
    def train_examples_and_features(self, train_examples, preprocessor):
        tes, tfs = [], []
        for te in train_examples:
            tep, tf = preprocessor.process_train(te)
            tes.append(tep)
            tfs.append(tf)
        return tes, tfs

    @pytest.fixture(scope='session')
    def eval_examples_and_features(self, eval_examples, preprocessor):
        ees, efs = [], []
        for ee in eval_examples:
            eep, ef = preprocessor.process_eval(ee)
            ees.append(eep)
            efs.append(ef)
        return ees, efs

    @pytest.fixture(scope='function')
    def trainer_with_mskd_extractive_model(
            self, training_args, config_and_model_with_extractive_head,
            tokenizer, train_examples_and_features, eval_examples_and_features, data_collator):
        _, model = config_and_model_with_extractive_head
        _, train_features = train_examples_and_features
        eval_examples, eval_features = eval_examples_and_features

        assert len(train_features) == 3
        assert len(eval_examples) == 3
        assert len(eval_features) == 3

        kd_teacher_dir = '/dccstor/arafat5/primeqa/primeqa/examples/domain-generalization-with-kd/models/CDLPV/bert-large-uncased/erm/checkpoint-108170'
        kd_teacher_model_path = os.path.join(kd_teacher_dir, 'pytorch_model.bin')
        kd_teacher_config_path = os.path.join(kd_teacher_dir, 'config.json')
        
        # Add distillation arguments to training_args        
        if model.name_or_path in ['bert-base-uncased', 'bert-large-uncased'] and os.path.exists(kd_teacher_model_path) and os.path.exists(kd_teacher_config_path):
            kd_args = {
                'kd_teacher_model_path': kd_teacher_model_path,
                'kd_teacher_config_path': kd_teacher_config_path,
                'kd_temperature': 2,
                'task_heads': EXTRACTIVE_HEAD
            }
        else:
            kd_args = {
                'kd_teacher_model_path': None,
                'kd_teacher_config_path': None,
                'kd_temperature': None
            }
        for k in kd_args:
            setattr(training_args, k, kd_args[k])

        # Simulate training on n=3 datasets and evaluating on m=4 datasets
        train_features = ConcatDataset(train_features)
        eval_features = ConcatDataset(eval_features)

        return MSKD_MRCTrainer(
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

    def test_trainer_trains_and_infers_without_errors(self, trainer_with_mskd_extractive_model):
        train_result = trainer_with_mskd_extractive_model.train()
        assert isinstance(train_result.training_loss, float)

        eval_metrics = trainer_with_mskd_extractive_model.evaluate()
        assert eval_metrics == {}
