from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import os

ELASTIC_PASSWORD=os.getenv("ELASTIC_PASSWORD")

client = Elasticsearch(
    cloud_id="sap-deployment:dXMtZWFzdC0xLmF3cy5mb3VuZC5pbzo0NDMkOGYwZTRiNTBmZGI1NGNiZGJhYTk3NjhkY2U4N2NjZTAkODViMzExOTNhYTQwNDgyN2FhNGE0MmRiYzg5ZDc4ZjE=",
    basic_auth=("elastic", ELASTIC_PASSWORD)
)
source = {"index": "bm25-s4hana-business1-successfactors-full-2023-09-12", "size": 10000}
dest={"index": "bm25-to-elser-s4hana-business1-successfactors-full-2023-09-12",
      "pipeline": "elser-v1-test"}

aa = client.reindex(source=source,
               dest=dest,
               conflicts="proceed",
               wait_for_completion=False
               )
print(aa)