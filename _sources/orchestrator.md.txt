<!---
Copyright 2022 PrimeQA Team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

<!-- <div align="center">
    <h1>PrimeQA Orchestrator Service REST Microservice</h1>
    <img src="_static/img/PrimeQA.png" width="150">
    <p>Proof-of-concept code for PrimeQA orchestrator microservice with integration to IBM Watson Discovery and machine reading comprehension engines as a REST Server.</p>
</div> -->

<div align="center">
    <img src="_static/img/PrimeQA.png" width="150"/>
</div>

# Orchestrator Service REST Microservice             

Proof-of-concept code for PrimeQA orchestrator microservice with integration to IBM Watson Discovery and machine reading comprehension engines as a REST Server.        
<br>

[![LICENSE|Apache2.0](https://img.shields.io/github/license/saltstack/salt?color=blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)

<h3>✔️ Getting Started</h3>

- [Repository](https://github.ibm.com/IBM-Research-AI/playground)        
- [Demo](http://mnlp-qa-dev-2.sl.cloud9.ibm.com:50059/docs)

<h3>✅ Prerequisites</h3>

- Python 3.9
  - If you are not using a Python version manager, [pyenv](https://realpython.com/intro-to-pyenv/#installing-pyenv) is
    highly recommended

<h3>🧩 Setup Local Environment</h3>

- [Setup and activate a Virtual Environment](https://docs.python.org/3/tutorial/venv.html) (follow steps below) or use [Conda](https://docs.conda.io/en/latest/miniconda.html)

```shell
# Install virtualenv
pip3 install virtualenv

# Create a new virtual environment for this project. If using pyenv, path_to_python_3.9_executable will be ~/.pyenv/versions/3.9.x/bin/python
virtualenv --python=<path_to_python_3.9_executable> venv

# Activate virtual environment
source venv/bin/activate
```

- Install dependencies

```shell
pip install -r requirements.txt
pip install -r requirements_test.txt
```

<h3>📜 TLS and Certificate Management</h3>

Orchestrator service REST server supports mutual or two-way TLS authentication (also known as mTLS).  
Application [`config.ini`](../config/config.ini) file contains the default certificate paths, but they can be overridden using environment variables.
All certificates are added using volume mounts on the application container. They _are not_ shipped along with the Docker image.  
Self-signed certs are added for running unit tests and local development testing. They are present in [`/secutiry/certs`](../security/certs) directory.  
These certificates are valid for about a 100 years (until 2122) from when they were created but if you want to generate a new set of certificates, follow the steps below:

- Navigate to [`/scripts`](../scripts) directory
- Run `./generate-certs.sh`
- When prompted for DN fields, leave everything (Country, State, Locality, Org, Unit, Email) as blank but the Common Name
  (CN). The fields can be left blank by just pressing Enter or return key. Use the following CNs for CA, Server and Client
  certificates:
  - First will be the CA certificate; use `CA` as the CN
  - Second will be the Server certificate; use `localhost` as the CN
  - Third will be the Client certificate; use `Client` as the CN
- The [`/security/certs` directory is mounted to `/opt/tls`](../scripts/run-locally.sh) on the application container and by default, the application tries to load certs from `/opt/tls`.

<h3>💻 Run Locally</h3>

- Open Python IDE & set the created virtual environment
- Open `orchestrator/services/config/config.ini`, set `require_ssl = false` (if you don't use TLS authentication) & `rest_port`
- Open `application.py` and run/debug
- Go to <http://localhost:{rest_port}/docs>
- Execute `PATCH settings` service with the [`primeqa.json`](data/primeqa.json) file content   
- To be able to use all the services, be sure to have run the PrimeQA container
  - Open PrimeQA directory
  - Follow README to set it up & generate image
  - Run `docker run --rm --name primeqa -d -p 50051:50051 --mount type=bind,source=/data/primeqa/store,target=/store -e STORE_DIR=/store -e mode=grpc -e require_ssl=false primeqa:$(cat VERSION)`

<h3>💻 Setup & Run Docker</h3>

- Open `Dockerfile` and set `port`
- Open `config.ini` and set `rest_port`
- Run `docker build -f Dockerfile -t primeqa-orchestrator:$(cat VERSION) --build-arg image_version:$(cat VERSION) .` (creates docker image)
- Run `docker run --rm --name primeqa_orchestrator -d -p 50059:50059 --mount type=bind,source="$(pwd)"/store,target=/store -e STORE_DIR=/store -e require_ssl=false primeqa_orchestrator:$(cat VERSION)` (run docker container)
- Go to container exposed url:port `/docs`  
- Execute `PATCH settings` service with the [`primeqa.json`](data/primeqa.json) file content   
- To be able to use all the services, be sure to have run the PrimeQA container
  - Open PrimeQA directory
  - Follow README to set it up & generate image
  - Run `docker run --rm --name primeqa -d -p 50051:50051 --mount type=bind,source=/data/primeqa/store,target=/store -e STORE_DIR=/store -e mode=grpc -e require_ssl=false primeqa:$(cat VERSION)`

<h3>📓 Third-party dependencies</h3>

- [ColBERT repository](https://github.ibm.com/IBM-Research-AI/ColBERT/tree/service): Please refer to ColBERT repository (specifically service branch) for more details around setting and running a local instance of NeuralIR search engine.
- [Watson Discovery](https://cloud.ibm.com/): Follow instructions on IBM Cloud to configure Watson Discovery V2 service.