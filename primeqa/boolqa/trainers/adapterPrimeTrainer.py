import logging
from typing import Optional
import torch
from transformers import AdapterTrainer, Trainer
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union, Type
from torch import nn
from timeit import default_timer

from primeqa.mrc.trainers.mrc import MRCTrainer

logger = logging.getLogger(__name__)

#
# this file creates a number of new trainer class via mixins
# we implement as mixins so that we can apply this functionality to either Trainer or AdapterTrainer
#
# InstrumentedMixin wraps training_step(), prediction_step() member functions with a timer
# HiddenStateMixin wraps training_step(), prediction_step() with extra input that will cause forward to return 
#    intermediate (hidden state) embeddings
#
# at the end we define various subclasses of both Trainer, AdapterTrainer 
# with one or both of these Mixins enabled:
# 
# class OhsAdapterTrainer(HiddenStateMixin, AdapterTrainer):
# class InstrumentedAdapterTrainer(InstrumentedMixin, AdapterTrainer):
# class InstrumentedOhsAdapterTrainer(InstrumentedMixin, HiddenStateMixin, AdapterTrainer):
# class InstrumentedTrainer(InstrumentedMixin, Trainer):
# class InstrumentedOhsTrainer(InstrumentedMixin, HiddenStateMixin, Trainer):
# class OhsTrainer(HiddenStateMixin, Trainer):
#
# As a reminder, in python multiple inheritance, super() always refers to the next class
# in the method-resolution-order list (not necessarily the immediate base class),
# which is defined by the _order_ in which the base classes are
# mentioned in the class definition
# So InstrumentedMixin should be first if you want to time everything,
# and the original trainer class should be last
#
# Thus in InstrumentedOhsAdapterTrainer, the method training_step()
# is provided by InstrumentedMixin.  The super() in InstrumentedMixin calls the implementation
# provided by HiddenStateMixin, and the super() in HiddenStateMixin calls the original
# AdapterTrainer implementation
#   
 
    
class InstrumentedMixin:
    """ instrument the training_step and prediction_step methods for timing
    """    
    def __init__(self, *args, **kwargs):
        print('InstrumentedMixin __init__')
        super().__init__(*args,**kwargs)        
        self._train_timestamps=[]
        self._predict_timestamps=[]
        self._cuda_is_available = torch.cuda.is_available()

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
#        print('InstrumentedMixin training_step')
        ts1=default_timer()        
        tr_loss_step=super().training_step(model,inputs)
        if self._cuda_is_available:
             torch.cuda.synchronize()
        ts2=default_timer()
        self._train_timestamps.append( ts2-ts1 )
        return tr_loss_step

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        ts1=default_timer()
        loss, logits, labels=super().prediction_step(model,inputs,prediction_loss_only,ignore_keys)
        if self._cuda_is_available:        
            torch.cuda.synchronize()        
        ts2=default_timer()
        self._predict_timestamps.append( ts2-ts1 )
        return loss, logits, labels 
#----------------------------------------------------------------------------------------------------------------

class HiddenStateMixin:
    """  pass output_hidden_states=True to trainer class
    """
    def __init__(self, *args, **kwargs):
        print('HiddenStateMixin __init__')        
        super().__init__(*args,**kwargs)        

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """ 
#        print('HiddenStateMixin training_step')
        inputs['output_hidden_states']=True
        tr_loss_step=super().training_step(model,inputs)
        return tr_loss_step

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on :obj:`model` using obj:`inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to evaluate.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (:obj:`bool`):
                Whether or not to return the loss only.
            ignore_keys (:obj:`Lst[str]`, `optional`):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        inputs['output_hidden_states']=True     
        loss, logits, labels=super().prediction_step(model,inputs,prediction_loss_only,ignore_keys)
        return loss, logits, labels

#------------------------------------------------------------------------------------------

class OhsAdapterTrainer(HiddenStateMixin, AdapterTrainer):
    pass


class InstrumentedAdapterTrainer(InstrumentedMixin, AdapterTrainer):
    pass

class InstrumentedOhsAdapterTrainer(InstrumentedMixin, HiddenStateMixin, AdapterTrainer):
    pass

class InstrumentedTrainer(InstrumentedMixin, Trainer):
    pass

class InstrumentedOhsTrainer(InstrumentedMixin, HiddenStateMixin, Trainer):
    pass

class OhsTrainer(HiddenStateMixin, Trainer):
    pass

class InstrumentedMRCTrainer(InstrumentedMixin, MRCTrainer):
    pass
