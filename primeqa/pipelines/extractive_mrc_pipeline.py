from transformers import AutoConfig, AutoTokenizer
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor

from transformers import DataCollatorWithPadding
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers

from primeqa.mrc.trainers.mrc import MRCTrainer
from datasets import Dataset
import json

class MRCPipeline():
    
    def __init__(self,model_for_qa):
        task_heads = EXTRACTIVE_HEAD
        config = AutoConfig.from_pretrained(
            model_for_qa,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_for_qa,
            use_fast=True,
            config=config,
        )

        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = ModelForDownstreamTasks.from_config(
            config,
            model_for_qa,
            task_heads=task_heads,
        )
        model.set_task_head(next(iter(task_heads)))        

        self.preprocessor = BasePreProcessor(
            stride=128,
            tokenizer=tokenizer,)
        
        data_collator = DataCollatorWithPadding(tokenizer)
        postprocessor = ExtractivePostProcessor(
            k=3,
            n_best_size=20,
            max_answer_length=30,
            scorer_type=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF,
            single_context_multiple_passages=self.preprocessor._single_context_multiple_passages,
        )
        
        self.trainer = MRCTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocessor.process
        )
        
    def predict(self, question, context):
        questions = [question]
        contexts = [[context]]
        example_ids = [str(0)]
        
        examples_dict = dict(question=questions, context=contexts, example_id=example_ids)
        eval_examples = Dataset.from_dict(examples_dict)
        
        eval_examples, eval_dataset = self.preprocessor.process_eval(eval_examples)
        predictions = self.trainer.predict(eval_dataset=eval_dataset, eval_examples=eval_examples)
        
        original_answers = list(predictions.values())[0]
        
        processed_answers = []
        for a in original_answers:
            processed_answers.append({'span_answer_text' : a['span_answer_text'],
                                        'confidence_score': a['confidence_score']})
        return processed_answers