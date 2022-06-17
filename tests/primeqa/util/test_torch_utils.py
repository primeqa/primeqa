from primeqa.util.transformers_utils.optimizer_utils import LossHistory


class Tester:
    def test_loss_history(self):
        history = LossHistory(10)
        for _ in range(100):
            history.note_loss(1)

    def test_optimizer(self):
        # need a dummy model
        pass