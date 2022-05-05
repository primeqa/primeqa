from transformers import Trainer


class QGTrainer(Trainer):

    def __init__(self,*args,train_dataset,valid_dataset,data_collator,**kwargs):
        """ QG training and Evaluation

        Args:
            Trainer (_type_)): _description_
        """
    super().init(*args,train_Dataset=train_dataset,valid_Dataset=valid_dataset,data_collator=data_collator,**kwargs)




