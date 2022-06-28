from transformers import Seq2SeqTrainer


class QGTrainer(Seq2SeqTrainer):

    def __init__(self,*args,train_dataset,valid_dataset,data_collator,**kwargs):
        """Question Generation Trainer for training and Evaluation. This class extends the Trainer class from huggingface.

        Args:
            train_dataset (Dataset): Train Dataset Generator
            valid_dataset (Dataset): Validation Dataset Generator
            data_collator (Dataset): Test Dataset Generator
        """  
        super().__init__(*args,
                train_dataset=train_dataset,
                eval_dataset=valid_dataset,
                data_collator=data_collator,
                **kwargs)
