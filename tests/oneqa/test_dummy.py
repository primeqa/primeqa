import torch
import transformers

from oneqa.hello_world import hello_world


class TestDummy:
    def test_torch(self):
        _ = torch.__version__

    def test_transformers(self):
        _ = transformers.__version__

    def test_hello_world(self):
        assert "Hello, World!" == hello_world()