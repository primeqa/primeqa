from primeqa.util.transformers_utils.hypers_base import HypersBase


class Tester:
    def test_hypers(self):
        hypers = HypersBase()
        hypers.set_seed(123)
        hypers.from_dict({})
        hypers.kofn('1of2')
        try:
            hypers.set_gradient_accumulation_steps()
        except:
            pass  # NOTE: fails when no GPU
        hypers._post_init()
        try:
            hypers.fill_from_args()
        except:
            pass
