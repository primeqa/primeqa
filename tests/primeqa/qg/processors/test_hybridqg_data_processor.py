import pytest
from primeqa.qg.processors.hybrid_qg.hybridqa_processor import HybridQADataset


def test_preprocess_data_for_qg():
    hd = HybridQADataset()
    data = hd.preprocess_data_for_qg("validation[:100]")
    assert data
    assert len(data['question']) == len(data['input'])
