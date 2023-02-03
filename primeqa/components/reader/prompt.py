from typing import List
from dataclasses import dataclass, field
import json
from .LLMService import LLMService
    
from primeqa.components.base import Reader as BaseReader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

@dataclass
class PromptReader(BaseReader):
    
    def __post_init__(self):
        # Placeholder variables
        self._preprocessor = None
        self._trainer = None

    def __hash__(self) -> int:
        # Step 1: Identify all fields to be included in the hash
        hashable_fields = [
            k
            for k, v in self.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]

        # Step 2: Run
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v for k, v in vars(self).items() if k in hashable_fields }, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        pass

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        pass
    
    def create_prompt(self, 
                    questions: str,
                    contexts: List[str]) -> str:
        
        # Use the question and contexts to create a prompt
        
        return "We do not currently have a prompt"


@dataclass
class PromptGPTReader(PromptReader):
    api_key: str = field(
        metadata={"name": "The API key for OPENAI"},
    )
    model: str = field(
        default="text-davinci-003",
        metadata={"name": "Model"},
    )
    max_tokens: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    temperature: float = field(
        default=0.7, metadata={"name": "The temperature parameter used for generation"}
    )
    top_p: int = field(
        default=1, metadata={"name": "The top_p parameter used for generation"}
    )
    frequency_penalty: int = field(
        default=0,
        metadata={"name": "frequency_penalty"},
    )
    presence_penalty: int = field(
        default=0,
        metadata={"name": "presence_penalty"},
    )
    
    def eval(self, *args, **kwargs):
        pass
    
    def train(self, *args, **kwargs):
        pass
    
    def load(self, *args, **kwargs):
        pass
    
    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        *args,
        example_ids: List[str] = None,
        **kwargs,
    ):
        predictions = []
        for i,q in enumerate(questions):
            print(self.create_prompt(q,contexts[i]))
            predictions.append({'example_id':i, 'text':'This is a placeholder and we do not call the API'})
        return predictions

@dataclass
class PromptFLANReader(PromptReader):
    api_key: str = field(
        metadata={"name": "The API key for BAM"},
    )
    model_name: str = field(
        default="flan-t5-xxl",
        metadata={"name": "Model"},
    )
    max_tokens: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    temperature: float = field(
        default=0.7, metadata={"name": "The temperature parameter used for generation"}
    )
    top_p: int = field(
        default=1, metadata={"name": "The top_p parameter used for generation"}
    )
    frequency_penalty: int = field(
        default=0,
        metadata={"name": "frequency_penalty"},
    )
    presence_penalty: int = field(
        default=0,
        metadata={"name": "presence_penalty"},
    )
    use_bam: bool = field(
        default=True, metadata={"name": "if true, use bam to run FLAN-T5"}
    )

    model = None
    tokenizer = None
    
    def eval(self, *args, **kwargs):
        pass
    
    def train(self, *args, **kwargs):
        pass
    
    def load(self, *args, **kwargs):
        if self.use_bam:
            self.model = LLMService(token='pak-EsKayQ0iw6gj8Bw8JYbg3G3Ye_iMIqwop9aJlzRoz40', model_id="google/" + self.model_name)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained("google/" + self.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained("google/" + self.model_name)
    
    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        *args,
        example_ids: List[str] = None,
        **kwargs,
    ):
        predictions = []
        for i,q in enumerate(questions):
            if self.use_bam:
                r = self.model.generate([q], 256, 100)
                predictions.append({'example_id':i, 'text': r['results'][0]['generated_text']})
            else:
                inputs = self.tokenizer([q], return_tensors="pt")
                outputs = self.model.generate(**inputs)
                predictions.append({'example_id':i, 'text': self.tokenizer.batch_decode(outputs, skip_special_tokens=True)})
        return predictions


