from contextlib import redirect_stdout
import sys
from transformers import AutoConfig, AutoTokenizer
from primeqa.boolqa.score_normalizer.score_normalizer import ScoreNormalizer
from primeqa.boolqa.trainers.adapterPrimeTrainer import InstrumentedAdapterTrainer, InstrumentedTrainer, InstrumentedMRCTrainer
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from primeqa.mrc.models.heads.classification import CLASSIFICATION_HEAD, ClassificationHead
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD, ExtractiveQAHead
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor

from transformers import DataCollatorWithPadding
from primeqa.boolqa.processors.postprocessors.extractive import ExtractivePipelinePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1
from dataclasses import dataclass, field

import torch
from primeqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from primeqa.mrc.processors.preprocessors.tydiqa_google import TyDiQAGooglePreprocessor
from primeqa.mrc.trainers.mrc import MRCTrainer
import datasets
from datasets import Dataset, concatenate_datasets, load_dataset, DatasetDict
import json
from operator import attrgetter
from contextlib import redirect_stdout
from typing import Union, Optional, Dict, List, Type

from transformers import (
    AutoModelForSequenceClassification,
    HfArgumentParser, 
    TrainingArguments,
    AdapterConfig,
    AdapterTrainer,
    PretrainedConfig,
    default_data_collator
)
from primeqa.boolqa.processors.postprocessors.boolqa_classifier import BoolQAClassifierPostProcessor
from primeqa.boolqa.processors.preprocessors.boolqa_classifier import BoolQAClassifierPreProcessor
from primeqa.boolqa.processors.dataset.mrc2dataset import create_dataset_from_run_mrc_output, create_dataset_from_json_str
import pandas as pd
import numpy as np
from timeit import default_timer
from collections import defaultdict
import IPython


import logging
logger = logging.getLogger(__name__)




@dataclass
class TaskArguments:
    """
    Task specific arguments.
    """
    dataset_name: str = field(
        default='balanced',
        metadata={"help": "which experiment to do",
        "choices": ["balanced", "balanced-lh", "nq-all", "tydiqa", "factoid", "boolean", "list"]
                  }
    )
    adapters: bool = field(default = False)




class MRCPipeline():



    def __init__(self, training_args: TrainingArguments, composite_config_file : str, MRC_Preprocessor_class : Type, adapters : bool):
        self.timestamps=defaultdict(list)
        boolqa_config=json.load(open(composite_config_file))
        qtc_config=boolqa_config['qtc']
        evc_config=boolqa_config['evc']
        mrc_config=boolqa_config['mrc']
        sn_config=boolqa_config['sn']

        # NOTE: override the number of labels - we can't do this through config argument because its different for the different heads
        # so we dynamically create inner classes to override one parameter
        class ClassificationHead_qtc(ClassificationHead):
            def __init__(self, config: PretrainedConfig):
                super().__init__(config, num_labels_override=len(qtc_config['label_list']))

        class ClassificationHead_evc(ClassificationHead):
            def __init__(self, config: PretrainedConfig):
                super().__init__(config, num_labels_override=len(evc_config['label_list']))



        if adapters:
            task_heads = dict(qa_head=ExtractiveQAHead, qtc_head=ClassificationHead_qtc, evc_head=ClassificationHead_evc)                    
            self._setup_mrc_model( training_args, mrc_config, task_heads, MRC_Preprocessor_class)
            self._setup_qtc_model_adapters( training_args, qtc_config, self.mrc_model, self.tokenizer)
            self._setup_evc_model_adapters( training_args, evc_config, self.mrc_model, self.tokenizer)
            self.set_model_to_mrc=self.set_model_to_mrc_adapters
            self.set_model_to_qtc=self.set_model_to_qtc_adapters
            self.set_model_to_evc=self.set_model_to_evc_adapters
            self.unset_model_to_mrc=self.unset_model_to_mrc_adapters
            self.unset_model_to_qtc=self.unset_model_to_qtc_adapters
            self.unset_model_to_evc=self.unset_model_to_evc_adapters            
        else:
            task_heads = dict(qa_head=ExtractiveQAHead)
            self._setup_mrc_model( training_args, mrc_config, task_heads, MRC_Preprocessor_class)
            self.unset_model_to_mrc_no_adapters()                         
            task_heads = dict(qtc_head=ClassificationHead_qtc )
            self._setup_qtc_model_no_adapters( training_args, qtc_config, task_heads )
            task_heads = dict(evc_head=ClassificationHead_evc)                                
            self._setup_evc_model_no_adapters( training_args, evc_config, task_heads )
            self.set_model_to_mrc=self.set_model_to_mrc_no_adapters
            self.set_model_to_qtc=self.set_model_to_qtc_no_adapters
            self.set_model_to_evc=self.set_model_to_evc_no_adapters 
            self.unset_model_to_mrc=self.unset_model_to_mrc_no_adapters
            self.unset_model_to_qtc=self.unset_model_to_qtc_no_adapters
            self.unset_model_to_evc=self.unset_model_to_evc_no_adapters                        

        self._setup_sn_model(sn_config)

    def _setup_mrc_model(self, training_args: TrainingArguments, mrc_config : Dict, task_heads : List, MRC_Preprocessor_class : Type):
        model_for_mrc = mrc_config['model_name_or_path']
        config = AutoConfig.from_pretrained(
            model_for_mrc,
        )
        # tokenizer is the same for all models since xlmRoberta
        tokenizer = AutoTokenizer.from_pretrained(
            model_for_mrc,
            use_fast=True,
            config=config,
            model_max_length=512 # TODO - needed in adapterhub, not earlier?
        )

        #
        # set up the MRC model (assumed to be the base model for adapters)
        #
        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = ModelForDownstreamTasks.from_config(
            config,
            model_for_mrc,
            task_heads=task_heads
        )
        model.set_task_head('qa_head')


        preprocessor = MRC_Preprocessor_class(
            stride=128,
            tokenizer=tokenizer,
            load_from_cache_file=False,
        )

        
        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

        postprocessor = ExtractivePipelinePostProcessor(
            k=3,
            n_best_size=20,
            max_answer_length=30,
            scorer_type=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF,
            single_context_multiple_passages=preprocessor._single_context_multiple_passages,
        )
        
        trainer = InstrumentedMRCTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args
#            post_process_function=self.mrc_postprocessor.process
        )
        self.tokenizer=tokenizer
        self.mrc_model=model
        self.mrc_preprocessor = preprocessor
        self.mrc_postprocessor = postprocessor
        self.mrc_trainer = trainer




    def set_model_to_mrc_adapters(self):
        self.mrc_model.set_task_head('qa_head')
        self.mrc_model.set_active_adapters([])

    def set_model_to_qtc_adapters(self):
        self.mrc_model.set_task_head('qtc_head')
        self.mrc_model.set_active_adapters(['qtc'])

    def set_model_to_evc_adapters(self):
        self.mrc_model.set_task_head('evc_head')
        self.mrc_model.set_active_adapters(['evc'])

    def unset_model_to_mrc_adapters(self):
        pass

    def unset_model_to_qtc_adapters(self):
        pass

    def unset_model_to_evc_adapters(self):
        pass

    def set_model_to_mrc_no_adapters(self):
        pass
        #self.mrc_model.to(device='cuda')

    def set_model_to_qtc_no_adapters(self):
        pass
        #self.qtc_model.to(device='cuda')

    def set_model_to_evc_no_adapters(self):
        pass
        #self.evc_model.to(device='cuda')

    def unset_model_to_mrc_no_adapters(self):
        pass
        #self.mrc_model.cpu()

    def unset_model_to_qtc_no_adapters(self):
        pass
        #self.qtc_model.cpu()

    def unset_model_to_evc_no_adapters(self):
        pass
        #self.evc_model.cpu()


    def _setup_qtc_model_adapters(self, training_args: TrainingArguments, qtc_config : Dict, base_model, tokenizer):
        #
        # set up the question type classifier model
        #
        preprocessor = BoolQAClassifierPreProcessor(
            sentence1_key=qtc_config['sentence1_key'],
            sentence2_key=qtc_config['sentence2_key'],
            tokenizer=tokenizer,
            load_from_cache_file=False,
            max_seq_len=tokenizer.model_max_length,
            label_list=qtc_config['label_list'],
            id_key=qtc_config['id_key'],
            padding=False
        )


        qtc_adapter_config = AdapterConfig.load(
            qtc_config['model_name_or_path'] + '/adapter_config.json',
            non_linearity=None,  # non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=None  # reduction_factor=adapter_args.adapter_reduction_factor,
        )

        # TODO what if its not roberta?
        base_model.roberta.load_adapter(
            qtc_config['model_name_or_path'],            
            config=qtc_adapter_config,
            load_as="qtc",
        )

        head_wts=torch.load(qtc_config['model_name_or_path'] + '/pytorch_model_head.bin', map_location=torch.device('cpu'))
        # TODO this is ugly
        qtc_head_wts={
            'classifier.dense.weight': head_wts['task_heads.qa_head.classifier.dense.weight'],
            'classifier.dense.bias': head_wts['task_heads.qa_head.classifier.dense.bias'],
            'classifier.out_proj.weight': head_wts['task_heads.qa_head.classifier.out_proj.weight'],
            'classifier.out_proj.bias': head_wts['task_heads.qa_head.classifier.out_proj.bias']
        }
        base_model.task_heads.qtc_head.load_state_dict(qtc_head_wts)


        postprocessor = BoolQAClassifierPostProcessor(
            k=10, 
            drop_label=None,
            sentence1_key=qtc_config['sentence1_key'],
            label_list = qtc_config['label_list'],
            id_key=qtc_config['id_key'],
            output_label_prefix=qtc_config['output_label_prefix'],
        )

        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)
        base_model.roberta.set_active_adapters(['qtc'])
        trainer = InstrumentedAdapterTrainer( 
            model=base_model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args
        )
        self.qtc_postprocessor=postprocessor
        self.qtc_trainer=trainer
        self.qtc_label_list=qtc_config['label_list']
        self.qtc_preprocessor=preprocessor



    def _setup_qtc_model_no_adapters(self, training_args: TrainingArguments, qtc_config : Dict, task_heads : Dict):
        #
        # set up the question type classifier model
        #
        model_for_qtc = qtc_config['model_name_or_path']

        config = AutoConfig.from_pretrained(
            model_for_qtc,
        )
        # tokenizer is the same for all models since xlmRoberta
        tokenizer = AutoTokenizer.from_pretrained(
            model_for_qtc,
            use_fast=True,
            config=config,
            model_max_length=512 # TODO - needed in adapterhub, not earlier?
        )

        #
        # set up the MRC model (assumed to be the base model for adapters)
        #
        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = AutoModelForSequenceClassification.from_pretrained( model_for_qtc, config=config)

        preprocessor = BoolQAClassifierPreProcessor(
            sentence1_key=qtc_config['sentence1_key'],
            sentence2_key=qtc_config['sentence2_key'],
            tokenizer=tokenizer,
            load_from_cache_file=False,
            max_seq_len=tokenizer.model_max_length,
            label_list=qtc_config['label_list'],
            id_key=qtc_config['id_key'],
            padding=False
        )

        postprocessor = BoolQAClassifierPostProcessor(
            k=10, 
            drop_label=None,
            sentence1_key=qtc_config['sentence1_key'],
            label_list = qtc_config['label_list'],
            id_key=qtc_config['id_key'],
            output_label_prefix=qtc_config['output_label_prefix'],
        )

        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)
        trainer = InstrumentedTrainer( 
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args
        )
        self.qtc_model = model
        self.qtc_postprocessor=postprocessor
        self.qtc_trainer=trainer
        self.qtc_label_list=qtc_config['label_list']
        self.qtc_preprocessor=preprocessor
        self.unset_model_to_qtc_no_adapters()             



    def _setup_evc_model_no_adapters(self, training_args: TrainingArguments, evc_config : Dict, task_heads : Dict):
        #
        # set up the question type classifier model
        #
        model_for_evc = evc_config['model_name_or_path']

        config = AutoConfig.from_pretrained(
            model_for_evc,
        )
        # tokenizer is the same for all models since xlmRoberta
        tokenizer = AutoTokenizer.from_pretrained(
            model_for_evc,
            use_fast=True,
            config=config,
            model_max_length=512 # TODO - needed in adapterhub, not earlier?
        )

        #
        # set up the MRC model (assumed to be the base model for adapters)
        #
        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = AutoModelForSequenceClassification.from_pretrained( model_for_evc, config=config)

        preprocessor = BoolQAClassifierPreProcessor(
            sentence1_key=evc_config['sentence1_key'],
            sentence2_key=evc_config['sentence2_key'],
            tokenizer=tokenizer,
            load_from_cache_file=False,
            max_seq_len=tokenizer.model_max_length,
            label_list=evc_config['label_list'],
            id_key=evc_config['id_key'],
            padding="max_length"
        )

        postprocessor = BoolQAClassifierPostProcessor(
            k=10, 
            drop_label=evc_config['drop_label'],
            sentence1_key=evc_config['sentence1_key'],
            label_list = evc_config['label_list'],
            id_key=evc_config['id_key'],
            output_label_prefix=evc_config['output_label_prefix'],
        )

        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)
        trainer = InstrumentedTrainer( 
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args
        )
        self.evc_model = model
        self.evc_postprocessor=postprocessor
        self.evc_trainer=trainer
        self.evc_label_list=evc_config['label_list']
        self.evc_preprocessor=preprocessor
        self.unset_model_to_evc_no_adapters() 





    def _setup_evc_model_adapters(self, training_args: TrainingArguments, evc_config, base_model, tokenizer):
        #
        # now set up the evc classifier
        #
        preprocessor = BoolQAClassifierPreProcessor(
            sentence1_key=evc_config['sentence1_key'],
            sentence2_key=evc_config['sentence2_key'],
            tokenizer=tokenizer,
            load_from_cache_file=False,
            max_seq_len=tokenizer.model_max_length,
            label_list=evc_config['label_list'],
            id_key=evc_config['id_key'],
            padding=False
        )

        evc_adapter_config = AdapterConfig.load(
            evc_config['model_name_or_path'] + '/adapter_config.json',
            non_linearity=None,  # non_linearity=adapter_args.adapter_non_linearity,
            reduction_factor=None  # reduction_factor=adapter_args.adapter_reduction_factor,
        )

        base_model.roberta.load_adapter(
            evc_config['model_name_or_path'],            
            config=evc_adapter_config,
            load_as="evc",
        )

        head_wts=torch.load(evc_config['model_name_or_path'] + '/pytorch_model_head.bin', map_location=torch.device('cpu'))
        # TODO this is ugly
        evc_head_wts={
            'classifier.dense.weight': head_wts['task_heads.qa_head.classifier.dense.weight'],
            'classifier.dense.bias': head_wts['task_heads.qa_head.classifier.dense.bias'],
            'classifier.out_proj.weight': head_wts['task_heads.qa_head.classifier.out_proj.weight'],
            'classifier.out_proj.bias': head_wts['task_heads.qa_head.classifier.out_proj.bias']
        }
        base_model.task_heads.evc_head.load_state_dict(evc_head_wts)

        postprocessor = BoolQAClassifierPostProcessor(
            k=10, 
            drop_label=evc_config['drop_label'],
            sentence1_key=evc_config['sentence1_key'] ,           
            label_list = evc_config['label_list'],
            id_key=evc_config['id_key'],
            output_label_prefix=evc_config['output_label_prefix']
        )

        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

        evc_data_collator=default_data_collator
        base_model.roberta.set_active_adapters(['evc'])
        trainer = InstrumentedAdapterTrainer( 
            model=base_model,
            tokenizer=tokenizer,
            data_collator=evc_data_collator,
            args=training_args
        )
        self.evc_postprocessor=postprocessor
        self.evc_trainer=trainer
        self.evc_label_list=evc_config['label_list']
        self.evc_preprocessor=preprocessor
    
    def _setup_sn_model(self, sn_config : Dict):
        self.sn = ScoreNormalizer(sn_config['model_name_or_path'])
        self.sn.load_model()


    def _do_mrc_prediction(self, eval_examples : Dataset):
        ts1=default_timer()
        self.set_model_to_mrc()
        ts2=default_timer()
        eval_examples, eval_dataset = self.mrc_preprocessor.process_eval(eval_examples)
        ts3=default_timer()
        # later versions of trainer override predict so that it takes both eval_examples and eval_dataset, and calls the post-processor ...
        mrc_predictions = self.mrc_trainer.predict(eval_dataset)
        ts4=default_timer()
        mrc_eval_preds = self.mrc_postprocessor.process(eval_examples, eval_dataset, mrc_predictions.predictions)
        ts5=default_timer()
        self.unset_model_to_mrc()
        ts6=default_timer()
        self.timestamps['_do_mrc_predictions.all'].append(ts6-ts1)
        self.timestamps['_do_mrc_predictions.switch'].append(ts2-ts1)
        self.timestamps['_do_mrc_predictions.preprocess'].append(ts3-ts2)
        self.timestamps['_do_mrc_predictions.predict'].append(ts4-ts3)
        self.timestamps['_do_mrc_predictions.postprocess'].append(ts5-ts4)
        self.timestamps['_do_mrc_predictions.unswitch'].append(ts6-ts5)
        return mrc_eval_preds, eval_examples

    def _do_qtc_prediction(self, mrc_eval_preds):
        ts1=default_timer()
        self.set_model_to_qtc()
        ts2=default_timer()
        qtc_eval_examples=create_dataset_from_json_str(mrc_eval_preds, unpack=False)
        qtc_eval_examples=qtc_eval_examples.add_column('label',[self.qtc_label_list[0]])    # TODO why?
        qtc_eval_examples, qtc_eval_dataset = self.qtc_preprocessor.process_eval(qtc_eval_examples)
        ts3=default_timer()
        qtc_predictions = self.qtc_trainer.predict(qtc_eval_dataset, metric_key_prefix="predict").predictions
        ts4=default_timer()
        # TODO why process_references_and_predictions here - process doesn't return anything?
        qtc_eval_preds = self.qtc_postprocessor.process_references_and_predictions(qtc_eval_examples, qtc_eval_dataset, qtc_predictions).predictions
        ts5=default_timer()
        self.unset_model_to_qtc()
        ts6=default_timer()
        self.timestamps['_do_qtc_predictions.all'].append(ts6-ts1)
        self.timestamps['_do_qtc_predictions.switch'].append(ts2-ts1)
        self.timestamps['_do_qtc_predictions.preprocess'].append(ts3-ts2)
        self.timestamps['_do_qtc_predictions.predict'].append(ts4-ts3)
        self.timestamps['_do_qtc_predictions.postprocess'].append(ts5-ts4)
        self.timestamps['_do_qtc_predictions.unswitch'].append(ts6-ts5)
        return qtc_eval_preds



    def _do_evc_predictions(self, qtc_eval_preds):
        ts1=default_timer() 
        self.set_model_to_evc()
        ts2=default_timer()
        evc_eval_examples=create_dataset_from_json_str(qtc_eval_preds, unpack=False)
        evc_eval_examples=evc_eval_examples.remove_columns(['label'])    # TODO why?
        evc_eval_examples=evc_eval_examples.add_column('label',[self.evc_label_list[0]])
        evc_eval_examples, evc_eval_dataset = self.evc_preprocessor.process_eval(evc_eval_examples)
        ts3=default_timer()
        evc_predictions = self.evc_trainer.predict(evc_eval_dataset, metric_key_prefix="predict").predictions
        ts4=default_timer()
        evc_eval_preds = self.evc_postprocessor.process_references_and_predictions(evc_eval_examples, evc_eval_dataset, evc_predictions).predictions
        ts5=default_timer()
        self.unset_model_to_evc()
        ts6=default_timer()
        self.timestamps['_do_evc_predictions.all'].append(ts6-ts1)
        self.timestamps['_do_evc_predictions.switch'].append(ts2-ts1)
        self.timestamps['_do_evc_predictions.preprocess'].append(ts3-ts2)
        self.timestamps['_do_evc_predictions.predict'].append(ts4-ts3)
        self.timestamps['_do_evc_predictions.postprocess'].append(ts5-ts4)
        self.timestamps['_do_evc_predictions.unswitch'].append(ts6-ts5)
        return evc_eval_preds



    def _do_sn_predictions(self, eval_preds):
        ts1=default_timer()                        
        qa_pred_data = create_dataset_from_json_str(eval_preds, unpack=True)
        sn_pred_data = self.sn.normalize_scores_inner(qa_pred_data, 'boolean', 'no_answer')
        ts2=default_timer()
        self.timestamps['_do_sn_predictions'].append(ts2-ts1)                        
        return sn_pred_data

    # TODO deal with example_id
    def predict_one_question(self, question : str, context : str, example_id : str='0'):
        questions = [question]
        contexts = [context]
        example_ids = [example_id]

        # do machine reading comprehension
        # set_model_to_mrc(self.mrc_model) 
        if type(self.mrc_preprocessor)==BasePreProcessor:
            examples_dict = dict(question=questions, context=[contexts], example_id=example_ids)
        else:
            examples_dict = dict(question_text=questions, document_plaintext=contexts, example_id=example_ids)
        
        eval_examples = Dataset.from_dict(examples_dict)
        return self.predict(eval_examples)

    def predict(self, eval_examples : Dataset):
        mrc_eval_preds, mrc_eval_examples = self._do_mrc_prediction(eval_examples)
        example_id=next(iter(mrc_eval_preds))
        # do question type classification (prediction boolean/short_answer)
        qtc_eval_preds=self._do_qtc_prediction(mrc_eval_preds)


        # TODO for timing purposes its not conditional right now
        # do boolean answer classification (predict yes/no)
        #if question_type_pred=='boolean':
        if True:
            evc_eval_preds = self._do_evc_predictions(qtc_eval_preds)
            eval_preds=evc_eval_preds
        else:
            eval_preds=qtc_eval_preds
            boolean_answer_pred=None
            eval_preds[example_id][0]['boolean_answer_pred']='YES'

        # do score normalization
        # score normalization is regression, not adapter-transformer so no set_model here
        sn_pred_data=self._do_sn_predictions(eval_preds)

        
        span_answer_text=mrc_eval_preds[example_id][0]['span_answer_text']
        question_type_pred=qtc_eval_preds[example_id][0]['question_type_pred']
        boolean_answer_pred=eval_preds[example_id][0]['boolean_answer_pred']            
        confidence_score = sn_pred_data[0]['confidence_score']
        print(span_answer_text, question_type_pred, boolean_answer_pred, confidence_score)
        print(f"after qtc torch memory allocated {torch.cuda.memory_allocated()} \
            max memory {torch.cuda.max_memory_allocated()}")              
        return sn_pred_data, mrc_eval_examples, mrc_eval_preds, span_answer_text, question_type_pred, boolean_answer_pred, confidence_score, eval_preds


    def _write_timings(self, outfile ):
        # times in trainer.predict (gpu time)
        mrc_ts=pd.Series(self.mrc_trainer._predict_timestamps)
        qtc_ts=pd.Series(self.qtc_trainer._predict_timestamps)
        evc_ts=pd.Series(self.evc_trainer._predict_timestamps)

        # times counting preprocessing, etc
        mrc_pts=pd.Series(self.timestamps['_do_mrc_predictions.all'], dtype=np.float64)
        qtc_pts=pd.Series(self.timestamps['_do_qtc_predictions.all'], dtype=np.float64)
        evc_pts=pd.Series(self.timestamps['_do_evc_predictions.all'], dtype=np.float64)

        with open(outfile, 'wt') as out:
            with redirect_stdout(out):

                for key, ts in self.timestamps.items():
                    print(key, ':', pd.Series(ts, dtype=np.float64).describe(), '\n')

                print('-----------------------------------------------')
                print('mrc predict times: ' + str(mrc_ts.describe()) + '\n')
                print('mrc predict+process times: '+ str(mrc_pts.describe()))

                print('qtc predict times: ' + str(qtc_ts.describe()) + '\n') 
                print('qtc predict+process times: '+ str(qtc_pts.describe()))           

                print('evc predict times: ' + str(evc_ts.describe()) + '\n')    
                print('evc predict+process times: '+ str(evc_pts.describe()))           

                print('mrc total: ', sum(mrc_pts))
                print('qtc total: ', sum(qtc_pts))
                print('evc total: ', sum(evc_pts))


                print('mrc overhead: ', (sum(mrc_pts) - sum(mrc_ts)) / sum(mrc_pts) if sum(mrc_pts)>0 else 0)
                print('qtc overhead: ', (sum(qtc_pts) - sum(qtc_ts)) / sum(qtc_pts) if sum(qtc_pts)>0 else 0)
                print('evc overhead: ', (sum(evc_pts) - sum(evc_ts)) / sum(evc_pts) if sum(evc_pts)>0 else 0)



def main_1doc():
    global MRC_Preprocessor_class
    parser = HfArgumentParser(( TrainingArguments)) # to get fp16
    (training_args,) = parser.parse_args_into_dataclasses()


    MRC_Preprocessor_class=BasePreProcessor
    config_file='examples/boolqa/tydi_boolqa_config_adapters1.json'
    mrcPipeline = MRCPipeline( training_args, config_file )
    question='Do zebra finches have stripes?'
    passage='The morphological differences between the subspecies include differences in size. T. g. guttata is smaller than T. g. castanotis. In addition, the T. g. guttata males do not have the fine barring found on the throat and upper breast and have smaller breast bands.[10]'

    res=mrcPipeline.predict_one_question(question,passage)

    print('mrc predict times: ' + str(pd.Series(mrcPipeline.mrc_trainer._predict_timestamps).describe()) + '\n')
    print('qtc predict times: ' + str(pd.Series(mrcPipeline.qtc_trainer._predict_timestamps).describe()) + '\n')    
    print('evc predict times: ' + str(pd.Series(mrcPipeline.evc_trainer._predict_timestamps).describe()) + '\n')    

    for key, ts in mrcPipeline.timestamps.items():
        print(f'{key} process+predict times: ' + str(pd.Series(ts).describe()))





def main_tydi():
    parser = HfArgumentParser(( TrainingArguments, TaskArguments)) # to get fp16
    (training_args, task_args) = parser.parse_args_into_dataclasses()
    num_samples=None
    if task_args.adapters:
        config_file='examples/boolqa/tydi_boolqa_config_adapters1.json'
    else:
        config_file='examples/boolqa/tydi_boolqa_config.json'
    #dataset_name='tydiqa'
    #dataset_name='balanced'
    #eval_metrics = getattr(sys.modules[__name__], 'TyDiF1')()
    eval_metrics = TyDiF1()

    dataset_name = task_args.dataset_name

    if dataset_name=='tydiqa':
        dataset_config_name='primary_task'
        raw_datasets = datasets.load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=None
            )  
        ds=raw_datasets['validation']
        MRC_Preprocessor_class=TyDiQAPreprocessor


    if dataset_name=='balanced':
        raw_datasets=load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/1/balanced/nq-dev-tydiformat-balanced.json'])
        ds=raw_datasets['train']
        MRC_Preprocessor_class=TyDiQAGooglePreprocessor

    if dataset_name=='factoid':
        raw_datasets=load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/1/balanced/nq-dev-tydiformat-balanced-factoid.json'])
        ds=raw_datasets['train']
        MRC_Preprocessor_class=TyDiQAGooglePreprocessor
 
    if dataset_name=='boolean':
        raw_datasets=load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/1/balanced/nq-dev-tydiformat-balanced-boolean.json'])
        ds=raw_datasets['train']
        MRC_Preprocessor_class=TyDiQAGooglePreprocessor

    if dataset_name=='list':
        raw_datasets=load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/1/balanced/nq-dev-tydiformat-balanced-list.json'])
        ds=raw_datasets['train']
        MRC_Preprocessor_class=TyDiQAGooglePreprocessor        

    if dataset_name=='balanced-lh':
        raw_datasets=load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/1/balanced/nq-dev-tydiformat-balanced-langhack.json'])
        ds=raw_datasets['train']
        MRC_Preprocessor_class=TyDiQAGooglePreprocessor        

    if dataset_name=='nq-all':
        raw_datasets=load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/1/nq-all/nq-dev-tydiformat-00.jsonl.gz','/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/0/nq-dev-all/nq-dev-tydiformat-01.jsonl.gz'])
        ds=raw_datasets['train']
        MRC_Preprocessor_class=TyDiQAGooglePreprocessor    

    #raw_datasets=load_dataset('json',data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/0/balanced_t0.json/validation/'])

    #raw_datasets=DatasetDict.load_from_disk('/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/0/balanced/')
    

    #raw_datasets = load_dataset('json', data_files=['/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/nq/dev/nq-dev-tydiformat-00.jsonl.gz',
    #                                                '/dccstor/jsmc-nmt-01/bool/expts/adapters-pqa/balanced/nq/dev/nq-dev-tydiformat-01.jsonl.gz'],
    #                            cache_dir=None)



    mrcPipeline = MRCPipeline( training_args, config_file, MRC_Preprocessor_class, task_args.adapters )  
    if num_samples is not None:
        ds=ds.select( range(num_samples) )
    



    eval_predictions_processed=[]
    eval_examples_list=[]
    for iDoc in range(len(ds)):
        print(f'document number {iDoc}')
        preds=mrcPipeline.predict(ds.select([iDoc]))
        eval_predictions_processed.append(preds[0][0])
        eval_examples_list.append(preds[1])


    mrcPipeline._write_timings(f'{training_args.output_dir}/times')
    eval_examples=concatenate_datasets(eval_examples_list)

    with open(f'{training_args.output_dir}/eval_predictions_processed.json', 'wt') as out:
        json.dump(eval_predictions_processed,out)

    
#    eval_metrics = getattr(sys.modules[__name__], 'TyDiF1')()
    references = mrcPipeline.mrc_postprocessor.prepare_examples_as_references(eval_examples)

    # TODO ugly hack here - its tricky to override the behavior of metrics classes
    sys.stdout = open(f'{training_args.output_dir}/prettyprint_results.json', 'wt')
    boolean_eval_metric = eval_metrics.compute(predictions=eval_predictions_processed, references=references)
    with open(f'{training_args.output_dir}/all_results.json', 'wt') as f:
        json.dump(boolean_eval_metric, f, indent=4, sort_keys=True)        

#---------------------------------------------------------------------------
# extras added here
class MRCPipeline_noargs(MRCPipeline):
    def __init__(self, 
         composite_config_file : str, 
         MRC_Preprocessor_class : Type, 
         adapters : bool):
        parser = HfArgumentParser(( TrainingArguments)) # to get fp16
        (training_args,) = parser.parse_args_into_dataclasses()
        super().__init__(training_args, composite_config_file, MRC_Preprocessor_class, adapters)
#---------------------------------------------------------------------------

# do main
if __name__=='__main__':
    #main_1doc()
    main_tydi()