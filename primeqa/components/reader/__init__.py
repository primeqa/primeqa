from dataclasses import dataclass, field
from primeqa.components.reader.prompt import PromptReader,PromptGPTReader
from typing import List, Dict

@dataclass
class GenerativeReader():
    api_key: str = field(
        metadata={"name": "The API key for BAM https://bam.res.ibm.com/"}, default=None
    )
    model_type: str = field(
        default="HuggingFace",
        metadata={"name": "Model type: HuggingFace, OpenAI"},
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
    top_k: int = field(
        default=5, metadata={"name": "The top_p parameter used for generation"}
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
        if self.model_type == "HuggingFace":
            self.reader = PromptReader(model_name=self.model_name, 
                                       max_new_tokens=self.max_new_tokens,
                                       min_new_tokens=self.min_new_tokens,
                                       temperature=self.temperature,
                                       top_p=self.top_k,
                                       frequency_penalty=self.frequency_penalty,
                                       presence_penalty=self.presence_penalty)
        elif self.model_type == "OpenAI":
            self.reader = PromptGPTReader(api_key=self.api_key,
                                       model_name=self.model_name, 
                                       max_new_tokens=self.max_new_tokens,
                                       temperature=self.temperature,
                                       top_p=self.top_k,
                                       frequency_penalty=self.frequency_penalty,
                                       presence_penalty=self.presence_penalty)
        else:
            raise Exception(f"The model {self.model_type} is not in the list of supported models: HuggingFace, OpenAI")
        self.reader.load()
        
        
    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]]=None,
        example_ids: List[str] = None,
        *args,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        return self.reader.predict(questions,
                                    contexts=contexts,
                                    example_ids=example_ids,
                                    *args,
                                    *kwargs)