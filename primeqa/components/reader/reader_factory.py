from dataclasses import dataclass, field
from primeqa.components.reader.prompt import PromptReader,PromptGPTReader
from primeqa.components.reader.generative import GenerativeFiDReader
from typing import List, Dict
from transformers import AutoConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES

PRIMEQA_GENERATIVE_MODELS = ["BartFiDModelForDownstreamTasks", "T5FiDModelForDownstreamTasks"]

@dataclass
class GenerativeReader():
    '''
    This class will initialize the correct generative reader component based on the model type and model name.
    
    Parameters:
        model_type (str): Model type: HuggingFace, OpenAI
        model_name (str): The model, default google/flan-t5-large
        api_key (str): The API key for OpenAI
        max_new_tokens (int): Maximum length of question and context inputs to the model (in word pieces/bpes)
        min_new_tokens (int): Minimum new tokens that must be generated (in word pieces/bpes
        temperature (float): The temperature parameter
        top_p (float): The top_p parameter
        top_k (float): The top_k parameter 
        frequency_penalty (int): Generation argument for OpenAI
        presence_penalty (int): eneration argument for OpenAI
        
    Returns:
        reader (Reader): the generative reader innitialized by the class
    '''
    api_key: str = field(
        metadata={"name": "The API key for OpenAI"}, default=None
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
            config = AutoConfig.from_pretrained(self.model_name)
            if config.architectures[0] in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values() or \
                config.architectures[0] in MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES.values() :
                self.reader = PromptReader(model_name=self.model_name, 
                                        max_new_tokens=self.max_new_tokens,
                                        min_new_tokens=self.min_new_tokens,
                                        temperature=self.temperature,
                                        top_p=self.top_p,
                                        frequency_penalty=self.frequency_penalty,
                                        presence_penalty=self.presence_penalty)
            elif config.architectures[0]  in PRIMEQA_GENERATIVE_MODELS:
                self.reader = GenerativeFiDReader(model=self.model_name,
                                                 max_answer_length=self.max_new_tokens,
                                                 generation_max_length=self.max_new_tokens)
            else:
                raise Exception(f"The model {self.model_name} is not in the list of supported HuggingFace models")
        elif self.model_type == "OpenAI":
            self.reader = PromptGPTReader(api_key=self.api_key,
                                       model_name=self.model_name, 
                                       max_new_tokens=self.max_new_tokens,
                                       temperature=self.temperature,
                                       top_p=self.top_p,
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
        '''
            Parameters:
                questions (List[str]): List of questions to be answered
                contexts (List[List[str]]): List of contexts for every question
                example_ids (List[str]): A list of question ids, when available. Default None
            Returns:
                the predictions from the reader component
        '''
        return self.reader.predict(questions,
                                    contexts=contexts,
                                    example_ids=example_ids,
                                    *args,
                                    *kwargs)