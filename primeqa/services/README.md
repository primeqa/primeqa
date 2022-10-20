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

PrimeQA service gRPC and REST server supports mutual or two-way TLS authentication (also known as mTLS). Application's [`config.ini`](./config/config.ini) file contains the default certificate paths, but they can be overridden using environment variables.

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
Please see the default values in config.ini [here](./config/config.ini). These can be overridden using environment variables.

- By default, the service starts as a `grpc` service. Set the <b>mode</b> to `rest` to start as a REST server. 
- By default, `require_ssl` is set to false.
- Set the `grpc_port` and/or `rest_port` to a free port number.

<h3>💻 Local</h3> 

- Update config [here](./config/config.ini).
- Open [application.py](./application.py) and run/debug

This will start a `ReaderService`, a `IndexerService` and a `RetrieverService` and the following lines will be displayed:

```
{"time":"2022-10-20 12:14:01,814", "name": "ReaderService", "level": "INFO", "message": "ReaderService is successfully initialized."}
{"time":"2022-10-20 12:14:01,815", "name": "IndexerService", "level": "INFO", "message": "IndexerService is successfully initialized."}
{"time":"2022-10-20 12:14:01,815", "name": "RetrieverService", "level": "INFO", "message": "RetrieverService is successfully initialized."}
I1020 12:14:01.815917763 2539136 socket_utils_common_posix.cc:353] TCP_USER_TIMEOUT is available. TCP_USER_TIMEOUT will be used thereafter
{"time":"2022-10-20 12:14:01,817", "name": "GrpcServer", "level": "INFO", "message": "Server instance started on port 50055 - initialization took 0 seconds"}
```
- Use one of the [Clients](./Clients) to send requests to the service.

<h3>💻 Docker</h3>

Please verify if Docker is properly setup with `docker run hello-world`

<h4> Build Docker Image </h4>

```
docker build -f Dockerfiles/Dockerfile.cpu -t primeqa:$(cat VERSION) --build-arg image_version:$(cat VERSION) .
```
<h4> Run Container </h4>

The container needs write access to a cache directory for caching Huggingface model and datasets.  Additionally will need write access to a store directory for index creation. 

```
chmod -R 777 $HOME/.cache/
chmod -R 777 $PWD/store/
```

To start a `gRPC` service, run the following command, replace `<host-port>` with a free port number:
`
docker run --rm --name primeqa -it -p <host-port>:50051 --mount type=bind,source="$(pwd)"/store,target=/store --mount type=bind,source="$HOME"/.cache/huggingface/,target=/cache/huggingface/ -e STORE_DIR=/store -e mode=grpc -e require_ssl=false primeqa:$(cat VERSION)
`

To start a `rest` service, run the following command, replace `<host-port>` with a free port number:

`
docker run --rm --name primeqa -it -p <host-port>:50052 --mount type=bind,source="$(pwd)"/store,target=/store --mount type=bind,source="$HOME"/.cache/huggingface/,target=/cache/huggingface/ -e STORE_DIR=/store -e mode=rest -e require_ssl=false primeqa:$(cat VERSION)
`

WARNING: The PrimeQA orchestrator and UI will only work with gRPC deployment without SSL. The parameter `require-ssl` must be set to `false`.

<h3>💻 Clients</h3>

<h4>Python</h4>
TODO: Refer to the PrimeQA Orchestrator for examples on how to make gRPC calls via python.

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
