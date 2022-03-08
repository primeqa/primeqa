import pytest
from transformers import AutoConfig

from oneqa.mrc.models.heads.extractive import ExtractiveQAHead
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.unit_test import UnitTest


class TestExtractiveQAHead(UnitTest):
    @UnitTest.PARAMETERIZE_MODEL_NAME
    def test_instantiation(self, model_name):
        config = AutoConfig.from_pretrained(model_name)
        head = ExtractiveQAHead(config)
        assert head.num_labels == config.num_labels

    def test_correct_number_of_classification_labels_when_using_default(self):
        model_name = 'roberta-base'
        config = AutoConfig.from_pretrained(model_name)
        head = ExtractiveQAHead(config)
        assert head.num_classification_head_labels == len(TargetType)

    def test_correct_number_of_classification_labels_when_overridden(self):
        model_name = 'roberta-base'
        num_classification_labels = 16
        config = AutoConfig.from_pretrained(model_name)
        head = ExtractiveQAHead(config, num_labels_override=num_classification_labels)
        assert head.num_classification_head_labels == num_classification_labels
