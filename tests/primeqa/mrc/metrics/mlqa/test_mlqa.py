import datasets
import pytest

from tests.primeqa.mrc.common.base import UnitTest
from primeqa.mrc.metrics.mlqa.mlqa import MLQA


class TestMLQA(UnitTest):
    @pytest.fixture(scope='session')
    def metric(self):
        return MLQA()
    
    @pytest.fixture(scope='session')
    def dataset_config_name(self):
        return 'mlqa.zh.en'

    @pytest.fixture(scope='session')
    def references_and_predictions(self):
        references = [
            dict(id='13f8eea6654690750d6c7ed262aa7345a92738c2', answers=dict(text=['宪法保障宗教自由'], answer_start=[18])),
            dict(id='2c4f008cd63348244e4553d1c529665ad18f5a4a', answers=dict(text=['伊斯兰教'], answer_start=[32])),
            dict(id='b08184972e38a79c47d01614aa08505bb3c9b680', answers=dict(text=['十支'], answer_start=[153])), 
            dict(id='36e2d9a94f3e08427a85e202229e50350abc80e1', answers=dict(text=['《毁灭战士3》'], answer_start=[0])),
            dict(id='faf513d17057c6bb95175c670ec98061b5cb2fce', answers=dict(text=['BFG 9000和电锯'], answer_start=[197])), 
            dict(id='b3fa3ec7514fab23baa7b630814e670a1418ee40', answers=dict(text=['《毁灭战士3》'], answer_start=[342])), 
            dict(id='a7888e1802b7ed59e0d6e4b6ff4348595f5d1f74', answers=dict(text=['赫尔辛基'], answer_start=[88])), 
            dict(id='9835becbfb3914e25253151e3c43fe0bfa2fecc4', answers=dict(text=['1993'], answer_start=[0])), 
            dict(id='040b6b9d4ceb3cc4f0981e59aabc47227232e371', answers=dict(text=['压力、经济环境，网路上的娱乐和资讯'], answer_start=[294])), 
            dict(id='9e9a108fdf106838bceeb3b4d9693e39931df705', answers=dict(text=['蚁酸'], answer_start=[51])) 
        ]
        predictions = [
            dict(id='13f8eea6654690750d6c7ed262aa7345a92738c2', prediction_text='宪法保障宗教自由'),
            dict(id='2c4f008cd63348244e4553d1c529665ad18f5a4a', prediction_text='伊斯兰'),
            dict(id='b08184972e38a79c47d01614aa08505bb3c9b680', prediction_text='十'),
            dict(id='36e2d9a94f3e08427a85e202229e50350abc80e1', prediction_text='《毁灭战士3'),
            dict(id='faf513d17057c6bb95175c670ec98061b5cb2fce', prediction_text='BFG 9000和电锯'),
            dict(id='b3fa3ec7514fab23baa7b630814e670a1418ee40', prediction_text='毁灭战士3'),
            dict(id='a7888e1802b7ed59e0d6e4b6ff4348595f5d1f74', prediction_text='赫尔辛基'),
            dict(id='9835becbfb3914e25253151e3c43fe0bfa2fecc4', prediction_text='1993'),
            dict(id='040b6b9d4ceb3cc4f0981e59aabc47227232e371', prediction_text='压力、经济环境，网路上的娱乐和资讯'),
            dict(id='9e9a108fdf106838bceeb3b4d9693e39931df705', prediction_text='蚁酸'),
        ]
        return dict(references=references, predictions=predictions)

    def test_instantiate_metric_from_class(self, metric):
        _ = metric

    def test_instantiate_metric_from_load_metric(self):
        from primeqa.mrc.metrics.mlqa import mlqa
        _ = datasets.load_metric(mlqa.__file__)

    def test_compute_metric(self, metric, references_and_predictions, dataset_config_name):
        metric.add_batch(**references_and_predictions)
        actual_metric_values = metric.compute(dataset_config_name = dataset_config_name)
        expected_metric_values = {"exact_match": 80.0, "f1": 95.23809523809524}
        assert actual_metric_values == expected_metric_values
