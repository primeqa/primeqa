from primeqa.qg.processors.passage_qg.squad_processor import SquadDataset
import pytest


def test_preprocess_data_for_qg():
    sd = SquadDataset()
    data = sd.preprocess_data_for_qg("validation")
    assert data!=None
    assert len(data['question']) == len(data['input'])