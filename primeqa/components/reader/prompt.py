from typing import List
from dataclasses import dataclass, field
import json
from .LLMService import LLMService
import openai
import sys
    
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
                    question: str,
                    contexts: List[str],
                    prefix: str) -> str:
        
        # Use the question and contexts to create a prompt
        if contexts == None or len(contexts) == 0:
            return f"{prefix} Question: {question}"
        else:
            passages = ", ".join(contexts)
            return f"{prefix} Question: {question} Text: {passages}"

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
        openai.api_key = self.api_key
    
    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        example_ids: List[str] = None,
        *args,
        **kwargs,
    ):
        predictions = []
        for i,q in enumerate(questions):
            prompt = self.create_prompt(q,contexts[i],**kwargs)
            #print(prompt)
            response = openai.Completion.create(
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty
            )
            if 'choices' in response and response['choices']:
                text = response.choices[0]['text']
            else:
                text = "Something went wrong with the GPT service"
            predictions.append({'example_id':i, 'text':text})
        return predictions

@dataclass
class PromptFLANT5Reader(PromptReader):
    api_key: str = field(
        metadata={"name": "The API key for BAM https://bam.res.ibm.com/"},
        default = None
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
    min_tokens: int = field(
     default=100,
        metadata={
            "name": "Min sequence length",
            "description": "Minimum new tokens that must be generated (in word pieces/bpes)",
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
        default=False, metadata={"name": "if true, use bam to run FLAN-T5"}
    )

    model = None
    tokenizer = None
    
    def eval(self, *args, **kwargs):
        pass
    
    def train(self, *args, **kwargs):
        pass
    
    def load(self, *args, **kwargs):
        if self.use_bam:
            self.model = LLMService(token=self.api_key, model_id="google/" + self.model_name)
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
        
        for i, q in enumerate(questions):
            prompt = self.create_prompt(q, contexts[i], **kwargs)
            len_prompt = len(prompt)
            
            #adjust for max sequence of Flan T5
            prompt = prompt + " Answer: "

            if len_prompt > 512:
                prompt = prompt[:len_prompt-507]

            if self.use_bam:
                r = self.model.generate([prompt], self.max_tokens, self.min_tokens)
                predictions.append({'example_id':i, 'text': r['results'][0]['generated_text']})
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(**inputs)
                predictions.append({'example_id':i, 'text': self.tokenizer.batch_decode(outputs, skip_special_tokens=True)})
        return predictions


class BAMReader(PromptReader):
    
    api_key: str = field(
        metadata={"name": "The API key for BAM https://bam.res.ibm.com/"},
    )
    model_name: str = field(
        default="flan-t5-xxl",
        metadata={"name": "Model"},
    )
    max_tokens: int = field(
        default=1024,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    min_tokens: int = field(
     default=100,
        metadata={
            "name": "Min sequence length",
            "description": "Minimum new tokens that must be generated (in word pieces/bpes)",
        },   
    )
    temperature: float = field(
        default=0, metadata={"name": "The temperature parameter used for generation"}
    )
    top_p: int = field(
        default=1, metadata={"name": "The top_p parameter used for generation"}
    )
    top_k: int = field(
        default=5, metadata={"name": "The top_p parameter used for generation"}
    )

    def __init__(self, args, **kwargs):
        self.model = None
        self.api_key = args.api_key
    
    def eval(self, *args, **kwargs):
        pass
    
    def train(self, *args, **kwargs):
        pass
    
    def load(self, *args, **kwargs):
        if kwargs['model'] is not None:
            self.model_name = kwargs['model']
        self.model = LLMService(token=self.api_key, model_id=self.model_name)
       
    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        *args,
        example_ids: List[str] = None,
        **kwargs,
    ):
        predictions = []
        
        for i, q in enumerate(questions):
            prompt = self.create_prompt(q, contexts[i], prefix=kwargs['prefix'])
            len_prompt = len(prompt)
            #adjust for max sequence of Flan T5
            prompt = prompt + " Answer: "

            if len_prompt > 1024:
                prompt = prompt[:1024]
 
            r = self.model.generate([prompt], 
                max_new_tokens=kwargs['max_tokens'], 
                min_new_tokens=kwargs['min_tokens'], 
                temperature=kwargs['temperature'], 
                top_k=kwargs['top_k'], 
                top_p=kwargs['top_p'])
            if "error" in r:
                print("Error running BAM service: ")
                print(r)
                sys.exit(0)
            predictions.append({'example_id':i, 'text': r['results'][0]['generated_text']})
           
        return predictions


            
            
