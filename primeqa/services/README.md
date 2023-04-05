<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.mrc

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 


# PrimeQA Services

This contains support for deploying PrimeQA capabiities such as retrieval and machine reading comprehension as microservices. Both gRPC and REST interfaces are supported. 

## Setup

Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

<h3>📜 TLS and Certificate Management</h3>

PrimeQA service gRPC and REST server supports mutual or two-way TLS authentication (also known as mTLS). Application's [config.ini](./config/config.ini) file contains the default certificate paths, but they can be overridden using environment variables.

Self-signed certificates are generated and packaged with the Docker build.
Self-signed certs _may be_ required for local development and testing. If you want to generate them, follow the steps below:

```shell
#!/usr/bin/env bash

# Make neccessary directories
mkdir -p security/
mkdir -p security/certs/
mkdir -p security/certs/ca security/certs/server security/certs/client

# Generate CA key and CA cert
openssl req -x509 -days 365 -nodes -newkey rsa:4096 -subj "/C=US/ST=New York/L=Yorktown Heights/O=IBM/OU=Research/CN=example.com" -keyout security/certs/ca/ca.key -out security/certs/ca/ca.crt

# Generate Server key (without passphrase) and Server cert signing request
openssl req -nodes -new -newkey rsa:4096 -subj "/C=US/ST=New York/L=Yorktown Heights/O=IBM/OU=Research/CN=example.com" -keyout security/certs/server/server.key -out security/certs/server/server.csr

# Sign Server cert
openssl x509 -req -days 365 -in security/certs/server/server.csr -CA security/certs/ca/ca.crt -CAkey security/certs/ca/ca.key -CAcreateserial -out security/certs/server/server.crt

# Generate Client key (without passphrase) and Client cert signing request
openssl req -nodes -new -newkey rsa:4096 -subj "/C=US/ST=New York/L=Yorktown Heights/O=IBM/OU=Research/CN=example.com" -keyout security/certs/client/client.key -out security/certs/client/client.csr

# Sign Client cert
openssl x509 -req -days 365 -in security/certs/client/client.csr -CA security/certs/ca/ca.crt -CAkey security/certs/ca/ca.key -CAserial security/certs/ca/ca.srl -out security/certs/client/client.crt

# Delete signing requests
rm -rf security/certs/server/server.csr
rm -rf security/certs/client/client.csr
```

**IMPORTANT:**
- By default, PrimeQA services are set up to run without SSL. We recommend that you set up certificats and update the config.ini and set `require_ssl` to `true`. The application tries to load certs from `/opt/tls`. You will need to update appropriate `tls_*` variables in [`config.ini`](./config/config.ini) during local use.

- We recommend to generate certificates with official signing authority and use them via volume mounts in the application container.

<h2>🛠 Build & Deployment </h2>
<h3>💻 Config</h3>
Please see the default values in [here](./config/config.ini). These can be overridden using environment variables.

- By default, the service starts as a `grpc` service. Set the <b>mode</b> to `rest` to start as a REST server. 
- By default, `require_ssl` is set to false.
- Set the `grpc_port` and/or `rest_port` to a free port number.

<h3>💻 Local</h3> 

- Update config [here](./config/config.ini).
- Open [application.py](./application.py) and run/debug

This will start a `ReaderService`, a `IndexerService`, a `RetrieverService`, a `RerankerService` and the following lines will be displayed:

gRPC service:
```
{"time":"2023-03-21 23:38:33,628", "name": "GrpcServer", "level": "INFO", "message": "Server instance started on port 50055 - initialization took 0 seconds"}
```

REST service:
```
INFO:     Uvicorn running on http://0.0.0.0:50056 (Press CTRL+C to quit)
{"time":"2023-03-21 23:39:48,024", "name": "uvicorn.error", "level": "INFO", "message": "Uvicorn running on http://0.0.0.0:50056 (Press CTRL+C to quit)"}
```
- Use one of the [Clients](#clients) to send requests to the service.

<h3>💻 Docker</h3>

Please verify if Docker is properly setup with `docker run hello-world`

<h4> Build Docker Image </h4>

```
docker build -f Dockerfiles/Dockerfile.cpu -t primeqa:$(cat VERSION) --build-arg image_version:$(cat VERSION) .
```
<h4> Run Container </h4>

The container needs write access to a `cache` directory e.g. `$HOME/.cache/` for caching Huggingface model and datasets.  Additionally, it will need write access to a `store` directory, e.g. `$PWD/store/` for custom models and index creation. 

See [Store](./store)

```
chmod -R 777 $HOME/.cache/
chmod -R 777 $PWD/store/
```

To start a `gRPC` service, run the following command, replace `<host-port>` with a free port number:

```
docker run --rm --name primeqa -it -p <host-port>:50051 --mount type=bind,source="$(pwd)"/store,target=/store --mount type=bind,source="$HOME"/.cache/huggingface/,target=/cache/huggingface/ -e STORE_DIR=/store -e mode=grpc -e require_ssl=false primeqa:$(cat VERSION)
```

To start a `rest` service, run the following command, replace `<host-port>` with a free port number:

```
docker run --rm --name primeqa -it -p <host-port>:50052 --mount type=bind,source="$(pwd)"/store,target=/store --mount type=bind,source="$HOME"/.cache/huggingface/,target=/cache/huggingface/ -e STORE_DIR=/store -e mode=rest -e require_ssl=false primeqa:$(cat VERSION)
```

WARNING: The PrimeQA orchestrator and UI will only work with gRPC deployment without SSL. The parameter `require-ssl` must be set to `false`.

<h3>💻 Store</h3>

The `primeqa store` provides a location for dropping in your own models, index and checkpoints. The store directory within the container is mounted at `/store`.  The directory structure is as follows:

```
store
├── checkpoints
│   └── <dense-ir-checkpoint-name>
│       └── <checkpoint-file>
├── indexes
│   └── <collection-name>
│       ├── documents.sqlite
│       ├── documents.tsv
│       ├── index
│       └── information.json
└── models
    └── <reader-model-name>
        ├── config.json
        ├── pytorch_model.bin
        └── tokenizer.json
```
### Drop in a Reader model 

Create a directory under `models` and copy `pytorch_model.bin`, `config.json` and `tokenizer.json` files into the direcotory.

### Drop in a Reranker model 

The [ColBERTReranker](../components/reranker/colbert_reranker.py) requires a ColBERT checkpoint/model file. 
Create a subdirectory under `checkpoints` and copy the checkpoint/model file in that subdirectory.

### Drop in a Dense IR index and checkpoint 

- Create a directory under `checkpoints` and copy the checkpoint file, e.g. a ColBERT dnn or DPR model file,  here.  

- Create a directory under indexes with a unique name for the collection `<collection-name>`. Place the following files in the directory:
  1. `documents.tsv` a tsv file contains the passages that were indexed. The format is `id\ttext\ttitle`.  
  2. `index` is a directory containing the index
  3. Create `documents.sqlite` by running the following python code from `indexes/<collection-name>` directory. This file is required to fetch the document text.

      ```
      from sqlitedict import SqliteDict
      import csv
      
      documents_tsv_file_path = "documents.tsv"
      documents_sqlite_file_path = "documents.sqlite"

      with open(documents_tsv_file_path, "r", encoding="utf-8") as documents_file, SqliteDict(
        documents_sqlite_file_path, tablename="documents"
      ) as documents_db:
          csv_reader = csv.DictReader(documents_file, fieldnames=["id", "text", "title"], delimiter="\t")
          next(csv_reader)
          for row in csv_reader:
              assert len(row) == 3 or len(row) == 2, f'Invalid .tsv record (has to contain 2 or 3 fields): {row}'
              documents_db[row["id"]] = {
                  "document_id": row["id"],
                  "text": row["text"],
                  "title": row["title"] if len(row) == 3 else None
              }
          # Commit to save documents_db
          documents_db.commit()
      ```

  4. Create the file `information.json` and add the following information into the file:

  ```
      {
        "index_id": "<collection-name>",
        "status": "READY",
        "configuration": {
          "engine_type": "<engine_type>",  # select one of "BM25", "ColBERT" or "DPR"
          "checkpoint": "<checkpoint>"     # Set this to the folder name where the checkpoint is stored.
        }
      }
  ```

  NOTE: `engine_type` is a now required for all Retrievers.  If you have an existing information.json file, please add this field. `checkpoint` is required for DPR and ColBERT Retrievers.

  ```
  - The index is now available for search

<h3 id="clients">💻 Clients</h3>

<h4>Python</h4> 

[PrimeQA Orchestrator](https://github.com/primeqa/primeqa-orchestrator) has example code on how to make gRPC calls via python


<h4>GUI</h4>

[BloomRPC](https://github.com/uw-labs/bloomrpc) is a decent GUI gRPC client.

<h4>REST</h4>

Go to http://localhost:{rest_port}/docs

Example CURL to send a request to the `ExtractiveReader`

```
curl -X 'POST' \
  'http://9.59.199.84:50057/answers' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "reader": {
    "reader_id": "ExtractiveReader"
  },
  "queries": [
    "How many of Warsaw'\''s inhabitants spoke Polish in 1933?"
  ],
  "contexts": [
    [
      "most diverse city in Poland, with significant numbers of foreign-born inhabitants. In addition to the Polish majority, there was a significant Jewish minority in Warsaw. According to the Russian census of 1897, out of the total population of 638,000, Jews constituted 219,000 (around 34% percent). Warsaw'\''s prewar Jewish population of more than 350,000 constituted about 30 percent of the city'\''s total population. In 1933, out of 1,178,914 inhabitants 833,500 were of Polish mother tongue. World War II changed the demographics of the city, and to this day there is much less ethnic diversity than in the previous 300 years of"
    ]
  ]
}'
```
Example output:
```
[
  [
    {
      "text": "833,500",
      "start_char_offset": 452,
      "end_char_offset": 459,
      "confidence_score": 0.704502638571937,
      "context_index": 0
    },
    {
      "text": "1,178,914 inhabitants 833,500",
      "start_char_offset": 430,
      "end_char_offset": 459,
      "confidence_score": 0.1990295438872201,
      "context_index": 0
    },
    {
      "text": "out of 1,178,914 inhabitants 833,500",
      "start_char_offset": 423,
      "end_char_offset": 459,
      "confidence_score": 0.09646781754084299,
      "context_index": 0
    }
  ]
]
```


<h2>Additional References</h2>
<h3>Getting started with Docker</h3>
If you are unfamiliar with Docker, you may want to take a look at:

- [Learn Docker in 12 Minutes (video)](https://www.youtube.com/watch?v=YFl2mCHdv24)
- [Learn Docker in 15 Minutes](https://medium.com/@vegiops/learn-docker-in-15-minutes-87c18cb84cbd)
- [Docker for beginners](https://docker-curriculum.com/)

<h3>Getting started with GRPC</h3>

Please take a look at [GRPC Introduction](https://grpc.io/docs/what-is-grpc/introduction/)
