from transformers import Trainer
class TableQATrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        """Basic tableQA trainer which extends huggingface's transformers Trainer class

        Args:
            eval_examples (_type_, optional): _description_. Defaults to None.
            post_process_function (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(*args, **kwargs)