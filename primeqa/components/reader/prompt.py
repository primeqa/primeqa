from typing import List, Dict
from dataclasses import dataclass, field
import sys
import logging
import json

import openai
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,AutoConfig,AutoModelForCausalLM
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES

from primeqa.components.base import Reader as BaseReader
from primeqa.components.reader.LLMService import LLMService

logger = logging.getLogger(__name__)


@dataclass
class PromptBaseReader(BaseReader):

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

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def create_prompt(
        self, question: str, contexts: List[str], prefix="", suffix="", **kwargs
    ) -> str:
        prompt = ""
        # Use the question and contexts to create a prompt
        if contexts == None or len(contexts) == 0:
            prompt = f"{prefix} Question: {question}"
        else:
            passages = ", ".join(contexts)
            prompt = f"{prefix} Question: {question} Text: {passages}"
        if suffix:
            prompt += " " + suffix + ":"
        return prompt

    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        *args,
        example_ids: List[str] = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        pass


@dataclass
class PromptGPTReader(PromptBaseReader):
    api_key: str = field(
        default=None,
        metadata={"name": "The API key for OPENAI"},
    )
    model_name: str = field(
        default="text-davinci-003",
        metadata={"name": "Model"},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    temperature: float = field(
        default=1, metadata={"name": "The temperature parameter used for generation"}
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

    def __post_init__(self):
        # Placeholder variables
        self._model = None

        # Default variable
        self._chat_models = ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]

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
        openai.api_key = self.api_key

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(
        self,
        questions: List[str],
        *args,
        contexts: List[List[str]] = None,
        example_ids: List[str] = None,
        **kwargs,
    ):
        predictions = {}
        for i, q in enumerate(questions):
            passages = None
            if contexts:
                passages = contexts[i]
            prompt = self.create_prompt(q, passages, **kwargs)

            if self.model_name in self._chat_models:
                response = openai.ChatCompletion.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    # temperature=self.temperature,
                    # max_tokens=self.max_new_tokens,
                    # top_p=self.top_p,
                    # frequency_penalty=self.frequency_penalty,
                    # presence_penalty=self.presence_penalty,
                )
                if "choices" in response and response["choices"]:
                    text = response.choices[0]["message"]["content"]
                else:
                    text = "Something went wrong with the GPT service"
              
            else:
                response = openai.Completion.create(
                    model=self.model_name,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_new_tokens,
                    top_p=self.top_p,
                    frequency_penalty=self.frequency_penalty,
                    presence_penalty=self.presence_penalty,
                )
                if "choices" in response and response["choices"]:
                    text = response.choices[0]["text"]
                else:
                    text = "Something went wrong with the GPT service"
            processed_prediction = {}
            processed_prediction["example_id"] = i
            processed_prediction["span_answer_text"] = text
            processed_prediction["confidence_score"] = 1
            predictions[i] = [processed_prediction]
        return predictions


@dataclass
class PromptFLANT5Reader(PromptBaseReader):
    api_key: str = field(
        metadata={"name": "The API key for BAM https://bam.res.ibm.com/"}, default=None
    )
    model_name: str = field(
        default="google/flan-t5-large",
        metadata={"name": "Model"},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    min_new_tokens: int = field(
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

    def __post_init__(self):
        # Placeholder variables
        self._model = None
        self._device = None
        self._tokenizer = None

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
        if self.use_bam:
            self._model = LLMService(token=self.api_key, model_id=self.model_name)
        else:
            self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            self._model = self._model.to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(
        self,
        questions: List[str],
        *args,
        contexts: List[List[str]] = None,
        example_ids: List[str] = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        predictions = {}

        for question_idx, question in enumerate(questions):
            passages = None
            if contexts:
                passages = contexts[question_idx]

            prompt = self.create_prompt(question, passages, **kwargs)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            span_answer_text = ""
            if self.use_bam:
                resp = self._model.generate(
                    [prompt], self.max_new_tokens, self.min_new_tokens
                )
                span_answer_text = resp["results"][0]["generated_text"]
            else:
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    min_length=self.min_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                )
                span_answer_text = self._tokenizer.batch_decode(
                    outputs, skip_special_tokens=True
                )[0]
            processed_prediction = {}
            processed_prediction["example_id"] = question_idx
            processed_prediction["span_answer_text"] = span_answer_text
            processed_prediction["confidence_score"] = 1
            predictions[question_idx] = [processed_prediction]
        return predictions


@dataclass
class BAMReader(PromptBaseReader):
    api_key: str = field(
        default=None,
        metadata={"name": "The API key for BAM https://bam.res.ibm.com/"},
    )
    model_name: str = field(
        default="google/flan-t5-xxl",
        metadata={"name": "Model"},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    min_new_tokens: int = field(
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

    def __post_init__(self):
        # Placeholder variables
        self._model = None
        self._tokenizer = None

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
        self._model = LLMService(token=self.api_key, model_id=self.model_name)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(
        self,
        questions: List[str],
        *args,
        contexts: List[List[str]] = None,
        example_ids: List[str] = None,
        **kwargs,
    ):
        predictions = {}
        max_sequence_length = 1024

        for question_idx, question in enumerate(questions):
            prompt = self.create_prompt(
                question=question, context=contexts[question_idx], **kwargs
            )
            inputs = self._tokenizer(prompt, return_tensors="pt")

            if len(inputs["input_ids"][0]) > max_sequence_length:
                prompt = (
                    self._tokenizer.decode(
                        self._tokenizer(
                            prompt,
                            truncation=True,
                            max_length=max_sequence_length
                            - len(self._tokenizer(kwargs["suffix"])["input_ids"]),
                        )["input_ids"],
                        skip_special_tokens=True,
                    )
                    + kwargs["suffix"]
                )

            resp = self._model.generate(
                [prompt],
                max_new_tokens=self.max_new_tokens,
                min_new_tokens=self.min_new_tokens,
                temperature=self.temperature,
                top_k=self.top_k,
                top_p=self.top_p,
            )
            if "error" in resp:
                logger.error("Error running BAM service: ")
                logger.error(resp)
                return None
            processed_prediction = {}
            processed_prediction["example_id"] = question_idx
            processed_prediction["span_answer_text"] = resp["results"][0][
                "generated_text"
            ]
            processed_prediction["confidence_score"] = 1
            predictions[question_idx] = [processed_prediction]

        return predictions

@dataclass
class PromptReader(PromptBaseReader):
    model_name: str = field(
        default="google/flan-t5-large",
        metadata={"name": "Model"},
    )
    max_new_tokens: int = field(
        default=256,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
        },
    )
    min_new_tokens: int = field(
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

    def __post_init__(self):
        # Placeholder variables
        self._model = None
        self._device = None
        self._tokenizer = None

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
        
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._config = AutoConfig.from_pretrained(self.model_name)
        if self._config.architectures[0] in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name)
        elif self._config.architectures[0] in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values():
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self._model = self._model.to(self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(
        self,
        questions: List[str],
        *args,
        contexts: List[List[str]] = None,
        example_ids: List[str] = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        predictions = {}

        for question_idx, question in enumerate(questions):
            passages = None
            if contexts:
                passages = contexts[question_idx]

            prompt = self.create_prompt(question, passages, **kwargs)
            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
            span_answer_text = ""
           
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                min_length=self.min_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            span_answer_text = self._tokenizer.batch_decode(
                outputs, skip_special_tokens=True
            )[0]
            
            processed_prediction = {}
            processed_prediction["example_id"] = question_idx
            processed_prediction["span_answer_text"] = span_answer_text
            processed_prediction["confidence_score"] = 1
            predictions[question_idx] = [processed_prediction]
        return predictions