from transformers import Trainer


class QGTrainer(Trainer):

    def __init__(self,*args,train_dataset,valid_dataset,data_collator,**kwargs):
        """ QG training and Evaluation

        Args:
            Trainer (_type_)): _description_
        """
        super().__init__(*args,train_dataset=train_dataset,eval_dataset=valid_dataset,data_collator=data_collator,**kwargs)




