from typing import List, Literal, Union, Dict, Any
from pydantic import BaseModel

from primeqa.services.constants import IndexStatus

#############################################################################################
#                       PipelineParameter
#############################################################################################
class Parameter(BaseModel):
    parameter_id: str
    name: Union[str, None] = None
    description: Union[str, None] = None
    type: Union[str, None] = None
    value: Union[int, float, bool, str, None] = None
    options: Union[List[int], List[float], List[bool], List[str], None] = None
    range: Union[List[int], List[float], None] = None


#############################################################################################
#                       Reader
#############################################################################################
class Reader(BaseModel):
    reader_id: str
    parameters: Union[List[Parameter], None] = None


#############################################################################################
#                       GetAnswersRequest
#############################################################################################
class GetAnswersRequest(BaseModel):
    reader: Reader
    queries: List[str]
    contexts: Union[List[List[str]], None] = None


#############################################################################################
#                       Answer
#############################################################################################
class Answer(BaseModel):
    text: str
    start_char_offset: int
    end_char_offset: int
    confidence_score: float
    context_index: int


#############################################################################################
#                       Retriever
#############################################################################################
class Retriever(BaseModel):
    retriever_id: str
    parameters: Union[List[Parameter], None] = None


#############################################################################################
#                       Reranker
#############################################################################################
class Reranker(BaseModel):
    reranker_id: str
    parameters: Union[List[Parameter], None] = None


#############################################################################################
#                       RetrieveRequest
#############################################################################################
class RetrieveRequest(BaseModel):
    retriever: Retriever
    index_id: str
    queries: List[str]
    
    
#############################################################################################
#                       Document
#############################################################################################
class Document(BaseModel):
    text: str
    document_id: Union[str, None] = None
    title: Union[str, None] = None


#############################################################################################
#                       Hit
#############################################################################################
class Hit(BaseModel):
    document: Document
    score: float


#############################################################################################
#                       RerankRequest
#############################################################################################
class RerankRequest(BaseModel):
    reranker: Reranker
    queries: List[str]
    hitsperquery: List[List[Hit]]

#############################################################################################
#                       Indexer
#############################################################################################
class Indexer(BaseModel):
    indexer_id: str
    parameters: Union[List[Parameter], None] = None


#############################################################################################
#                       GenerateIndexRequest
#############################################################################################
class GenerateIndexRequest(BaseModel):
    indexer: Indexer
    documents: List[Document]
    index_id: Union[str, None] = None
    metadata: Union[str, Dict[str, Any]] = None


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
    configuration: Dict[str, Any]
    metadata: Union[Dict[str, Any], None] = None
