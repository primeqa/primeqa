from typing import List, Union
from typing_extensions import Literal
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

<<<<<<< HEAD
=======
# get table answer new 
>>>>>>> 5836e49540f284e3f1f540157535aa9978391e3e
class GetTableAnswerRequest(BaseModel):
    queries: List[str]
    contexts: dict

#############################################################################################
#                       Answer
#############################################################################################
class Answer(BaseModel):
    text: str
    start_char_offset: int
    end_char_offset: int
    confidence_score: float
    context_index: int

<<<<<<< HEAD
=======
# get table answer new 
>>>>>>> 5836e49540f284e3f1f540157535aa9978391e3e
class TableAnswer(BaseModel):
    answers: dict

#############################################################################################
#                       Retriever
#############################################################################################
class Retriever(BaseModel):
    retriever_id: str
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
