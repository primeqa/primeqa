from typing import List, Literal, Union
from pydantic import BaseModel

from primeqa.services.constants import IndexStatus

#############################################################################################
#                       PipelineParameter
#############################################################################################
class PipelineParameter(BaseModel):
    parameter_id: str
    name: Union[str, None] = None
    type: Union[str, None] = None
    value: Union[int, float, bool, str]
    options: Union[List[bool], List[str]] = None
    range: Union[List[int], List[float], None] = None


#############################################################################################
#                       Pipeline
#############################################################################################
class Pipeline(BaseModel):
    pipeline_id: str
    name: Union[str, None] = None
    type: Union[str, None] = None
    description: Union[str, None] = None
    parameters: Union[List[PipelineParameter], None] = None
    metadata: Union[dict, None] = None


#############################################################################################
#                       ReaderQuery
#############################################################################################
class ReaderQuery(BaseModel):
    pipeline: Pipeline
    question: str
    passages: List[str]
    metadata: Union[dict, None] = None


#############################################################################################
#                       Answer
#############################################################################################
class Answer(BaseModel):
    text: str
    start_char_offset: int
    end_char_offset: int
    confidence_score: float
    passage_index: int
    metadata: Union[dict, None] = None


#############################################################################################
#                       RetrieverQuery
#############################################################################################
class RetrieverQuery(BaseModel):
    pipeline: Pipeline
    index_id: str
    query: str
    index_id: str
    metadata: Union[dict, None] = None


#############################################################################################
#                       Document
#############################################################################################
class Document(BaseModel):
    document_id: str
    text: str
    metadata: Union[dict, None] = None


#############################################################################################
#                       Hit
#############################################################################################
class Hit(BaseModel):
    document: Document
    score: float
    metadata: Union[dict, None] = None


#############################################################################################
#                       IndexRequest
#############################################################################################
class IndexRequest(BaseModel):
    pipeline: Pipeline
    documents: List[Document]
    index_id: Union[str, None] = None
    metadata: Union[dict, None] = None


#############################################################################################
#                       Index
#############################################################################################
class IndexInformation(BaseModel):
    index_id: str
    status: Literal[
        IndexStatus.READY,
        IndexStatus.INDEXING,
        IndexStatus.DOES_NOT_EXISTS,
        IndexStatus.CORRUPT,
    ]
    metadata: Union[dict, None] = None
