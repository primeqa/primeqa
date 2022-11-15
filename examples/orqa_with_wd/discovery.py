from typing import List, Union
from dataclasses import dataclass, field

from ibm_watson import DiscoveryV2, ApiException
from ibm_cloud_sdk_core.authenticators import (
    IAMAuthenticator,
    BearerTokenAuthenticator,
)

from primeqa.pipelines.components.base import RetrieverComponent

# Configure IBM Watson discovery service connection
# ACTIVE_ENDPOINT = None
# WDS = None

@dataclass
class WatsonDiscoveryRetriever(RetrieverComponent):
    """_summary_

    Args:
        endpoint (str): url to Discovery instance.
        api_key  (str): Watson Discovery Api Key
        project_id (str): Watson Discovery project id
        index_name (str): collection name
        max_num_documents (int, optional): Maximum number of retrieved document. Defaults to 5.

    Returns:
        _type_: _description_

    """
    
    endpoint: str = field(
        default=None,
        metadata={
            "name": "endpoint",
            "description": "service url",
            "api_support": True,
        },
    )
    
    api_key: str = field(
        default=None,
        metadata={
            "name": "API Key",
            "description": "API Key",
            "api_support": True,
        },
    )
    
    project_id: str = field(
        default=None,
        metadata={
            "name": "Project id",
            "description": "Project id",
            "api_support": True,
        },
    )
    

    index_name: str = field(
        default=None,
        metadata={
            "name": "index name aka collection id",
            "description": "collection id",
            "api_support": True,
        },
    )
    
    max_num_documents: int = field(
        default=5,
        metadata={
            "name": "Maximum number of retrieved documents",
            "range": [1, 100, 1],
            "api_support": True,
        },
    )

    def __post_init__(self):
        # Placeholder variables
        self._WDS = None
        self._collection_id = None
        
    def __hash__(self) -> int:
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v.default for k, v in self.__class__.__dataclass_fields__.items() if not 'exclude_from_hash' in v.metadata or not v.metadata['exclude_from_hash']}, sort_keys=True)}"
        )  

    def load(self, *args, **kwargs):
        self._WDS = DiscoveryV2(version="2020-08-30", authenticator=IAMAuthenticator(apikey=self.api_key))
        self._WDS.set_service_url(self.endpoint)
        self.get_collection_id()
        if self._collection_id == None:
            raise RuntimeError(f"Index not found {self.index_name}")

    def get_collection_id(self):
        collections = self._WDS.list_collections(project_id=self.project_id).get_result()["collections"]
        for collection in collections:
            if collection['name'] == self.index_name:
                self._collection_id = collection['collection_id']
                break
        

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        
        results = []
        
        for query in input_texts:
            query_hits = []
            hits = self._WDS.query(
                    project_id=self.project_id,
                    collection_ids=[self._collection_id],
                    natural_language_query=query,
                    count=self.max_num_documents,
                ).get_result()["results"]
        
        
            if hits:
                query_hits = [
                {
                    "text": hit["text"][0],
                    "search_score": hit['result_metadata']['confidence'],
                    "document_id": hit["document_id"]
                    if "document_id" in hit
                    else None,
                    "title": hit["title"]
                    if "title" in hit
                    else None
                }
                for hit in hits
                ]
                results.append(query_hits)
            else:
                results.append([])
                
        return results
    
    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass
