from typing import List
from dataclasses import dataclass, field
from transformers import AutoConfig, AutoTokenizer
from transformers import Seq2SeqTrainingArguments
from datasets import Dataset

from primeqa.pipelines.components.base import ReaderComponent
from primeqa.mrc.models.heads.generative import FID_HEAD
from primeqa.mrc.models.fid_task_model import FiDModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.eli5_fid import ELI5FiDPreprocessor
from primeqa.mrc.data_models.data_collator import FiDDataCollator
from primeqa.mrc.processors.postprocessors.eli5_fid import ELI5FiDPostProcessor
from primeqa.mrc.trainers.seq2seq_mrc import MRCSeq2SeqTrainer

@dataclass
class GenerativeReader(ReaderComponent):
    """_summary_

    Args:

    Returns:
        _type_: _description_
    """

    def __post_init__(self):
        # Placeholder variables
        self._preprocessor = None
        self._trainer = None

    def load(self, *args, **kwargs):
        pass

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        pass
    
@dataclass
class GenerativeFiDReader(ReaderComponent):
    
    model: str = field(
        default="PrimeQA/eli5-fid-bart-large-with-colbert-passages",
        metadata={"name": "Model"},
    )
    max_seq_len: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
            "range": [32, 512, 8],
        },
    )
    max_answer_length: int = field(
        default=256, metadata={"name": "Maximum answer length", "range": [2, 2000, 2]}
    )
    generation_max_length: int = field(
        default=256, metadata={"name": "Maximum answer length", "range": [2, 2000, 2]}
    )
    generation_num_beams: int = field(
        default=1, metadata={"name": "The number of beams for generation", "range": [1, 5, 1]}
    )
    num_contexts: int = field(
        default=3, metadata={"name": "The number of passages in the input", "range": [1, 10, 1]}
    )

    def __post_init__(self):
        # Placeholder variables
        self._preprocessor = None
        self._trainer = None
    
    def load(self, *args, **kwargs):
        task_heads = FID_HEAD
        # Load configuration for model
        config = AutoConfig.from_pretrained(self.model)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            config=config,
        )

        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = FiDModelForDownstreamTasks.from_config(
            config,
            self.model,
            task_heads=task_heads,
        )
        model.set_task_head(next(iter(task_heads)))

        # Initialize preprocessor
        self._preprocessor = ELI5FiDPreprocessor(
            stride=0,
            max_seq_len=self.max_seq_len,
            tokenizer=tokenizer,
            max_contexts=3,# self.num_contexts,
            max_answer_len=self.generation_max_length
        )

        data_collator = FiDDataCollator(tokenizer)
        
         # Initialize post processor
        postprocessor = ELI5FiDPostProcessor(
            k=0,
            max_answer_length=self.max_answer_length,
            scorer_type=None,
            single_context_multiple_passages=self._preprocessor._single_context_multiple_passages,
            tokenizer=tokenizer,
        )

        training_args = Seq2SeqTrainingArguments(
            do_eval=True,
            output_dir="tmp_trainer",
            per_device_eval_batch_size=1,
            predict_with_generate=True,
            generation_max_length=self.max_answer_length,
            generation_num_beams=self.generation_num_beams
        )

        self._trainer = MRCSeq2SeqTrainer(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocessor.process,
        )
        
    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        processed_context = []
        for single_context in context:
            processed_context.append([{"text":t} for t in single_context])
        predict_examples = Dataset.from_dict(
            dict(
                input=input_texts,
                passages=processed_context,
                id=[str(idx) for idx in range(len(input_texts))],
            )
        )

        predict_examples, predict_dataset = self._preprocessor.process_eval(predict_examples)

        # Run predict
        predictions = []
        for raw_prediction in self._trainer.predict(
            predict_dataset=predict_dataset, predict_examples=predict_examples
        ):
            processed_prediction = {}
            processed_prediction["example_id"] = raw_prediction['id']
            processed_prediction["text"] = raw_prediction['prediction_text']
            predictions.append(processed_prediction)
        
        return predictions