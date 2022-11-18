<!---
Copyright 2022 IBM Corp.

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

<h3 align="center">
    <img width="350" alt="primeqa" src="docs/_static/img/PrimeQA.png">
    <p>The prime repository for state-of-the-art Multilingual and Multimedia Question Answering research and development.</p>
</h3>

![Build Status](https://github.com/primeqa/primeqa/actions/workflows/primeqa-ci.yml/badge.svg)
[![LICENSE|Apache2.0](https://img.shields.io/github/license/saltstack/salt?color=blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![sphinx-doc-build](https://github.com/primeqa/primeqa/actions/workflows/sphinx-doc-build.yml/badge.svg)](https://github.com/primeqa/primeqa/actions/workflows/sphinx-doc-build.yml)   

PrimeQA is a public open source repository that enables researchers and developers to train state-of-the-art models for question answering (QA). By using PrimeQA, a researcher can replicate the experiments outlined in a paper published in the latest NLP conference while also enjoying the capability to download pre-trained models (from an online repository) and run them on their own custom data. PrimeQA is built on top of the [Transformers](https://github.com/huggingface/transformers) toolkit and uses [datasets](https://huggingface.co/datasets/viewer/) and [models](https://huggingface.co/PrimeQA) that are directly downloadable.


The models within PrimeQA supports End-to-end Question Answering. PrimeQA answers questions via 
- [Information Retrieval](https://github.com/primeqa/primeqa/tree/main/primeqa/ir): Retrieving documents and passages using both traditional (e.g. BM25) and neural (e.g. ColBERT) models
- [Multilingual Machine Reading Comprehension](https://huggingface.co/ibm/tydiqa-primary-task-xlm-roberta-large): Extract and/ or generate answers given the source document or passage.
- [Multilingual Question Generation](https://huggingface.co/PrimeQA/mt5-base-tydi-question-generator): Supports generation of questions for effective domain adaptation over [tables](https://huggingface.co/PrimeQA/t5-base-table-question-generator) and [multilingual text](https://huggingface.co/PrimeQA/mt5-base-tydi-question-generator).

Some examples of models (applicable on benchmark datasets) supported are :
- [Traditional IR with BM25](https://github.com/primeqa/primeqa/tree/main/primeqa/ir/) Pyserini
- [Neural IR with ColBERT, DPR](https://github.com/primeqa/primeqa/tree/main/primeqa/ir) (collaboration with [Stanford NLP](https://nlp.stanford.edu/) IR led by [Chris Potts](https://web.stanford.edu/~cgpotts/) & [Matei Zaharia](https://cs.stanford.edu/~matei/)).
Replicating the experiments that [Dr. Decr](https://huggingface.co/ibm/DrDecr_XOR-TyDi_whitebox) (Li et. al, 2022) performed to reach the top of the XOR TyDI leaderboard.
- [Machine Reading Comprehension with XLM-R](https://github.com/primeqa/primeqa/tree/main/primeqa/mrc): to replicate the experiments to get to the top of the TyDI leaderboard similar to the performance of the IBM GAAMA system. Coming soon: code to replicate GAAMA's performance on Natural Questions. 
- [Multimedia QA over news & movies](https://arxiv.org/abs/2112.10728): coming soon! to replicate the experiments run over multi-hop QA over images, text over variety of domains. Collaboration with [UIUC Blender lab](https://blender.cs.illinois.edu/).



## ✔️ Getting Started

### Installation
[Installation doc](https://primeqa.github.io/primeqa/installation.html)       

```shell
# cd to project root

# If you want to run on GPU make sure to install torch appropriately

# E.g. for torch 1.11 + CUDA 11.3:
pip install 'torch~=1.11.0' --extra-index-url https://download.pytorch.org/whl/cu113

# Install as editable (-e) or non-editable using pip, with extras (e.g. tests) as desired
# Example installation commands:

# Minimal install (non-editable)
pip install .

# GPU support
pip install .[gpu]

# Full install (editable)
pip install -e .[all]
```

Please note that dependencies (specified in [setup.py](./setup.py)) are pinned to provide a stable experience.
When installing from source these can be modified, however this is not officially supported.

**Note:** in many environments, conda-forge based faiss libraries perform substantially better than the default ones installed with pip. To install faiss libraries from conda-forge, use the following steps:

- Create and activate a conda environment
- Install faiss libraries, using a command

```conda install -c conda-forge faiss=1.7.0 faiss-gpu=1.7.0```

- In `setup.py`, remove the faiss-related lines:

```commandline
"faiss-cpu~=1.7.2": ["install", "gpu"],
"faiss-gpu~=1.7.2": ["gpu"],
```

- Continue with the `pip install` commands as desctibed above.


### JAVA requirements
Java 11 is required for BM25 retrieval. 

Download Java 11 package from https://jdk.java.net/archive/ and uncompress

Set JAVA_HOME:
```shell
export JAVA_HOME=<jdk-dir>
export PATH=$JAVA_HOME/bin:$PATH
```

## 🧪 Unit Tests
[Testing doc](https://primeqa.github.io/primeqa/testing.html)       

To run the unit tests you first need to [install PrimeQA](#Installation).
Make sure to install with the `[tests]` or `[all]` extras from pip.

From there you can run the tests via pytest, for example:
```shell
pytest --cov PrimeQA --cov-config .coveragerc tests/
```

For more information, see:
- Our [tox.ini](./tox.ini)
- The [pytest](https://docs.pytest.org) and [tox](https://tox.wiki/en/latest/) documentation    

## 🏅 Leaderboard

PrimeQA is at the top of several leaderboards: XOR-TyDi, TyDiQA-main, OTT-QA and HybridQA.

### [XOR-TyDi](https://nlp.cs.washington.edu/xorqa/)
<img src="https://user-images.githubusercontent.com/4257308/202567250-ef83e72a-88f0-416c-814a-80fea865e2a3.png" width="50%">

### [TyDiQA-main](https://ai.google.com/research/tydiqa)
<img src="https://user-images.githubusercontent.com/4257308/202567359-3c411ffa-8a27-4796-8f37-04ec74588242.png" width="50%">

### [OTT-QA](https://codalab.lisn.upsaclay.fr/competitions/7967)
<img src="https://user-images.githubusercontent.com/4257308/202567404-0fa060a0-8b28-4ac6-9e3a-7b021a707666.png" width="50%">

### [HybridQA](https://codalab.lisn.upsaclay.fr/competitions/7979)
<img src="https://user-images.githubusercontent.com/4257308/202567436-076dbe61-1482-4f02-865c-9d422ed503af.png" width="50%">


## 🔭 Learn more

| Section | Description |
|-|-|
| 📒 [Documentation](https://primeqa.github.io/primeqa) | Full API documentation and tutorials |
| 🏁 [Quick tour: Entry Points for PrimeQA](https://github.com/primeqa/primeqa/tree/main/primeqa) | Different entry points for PrimeQA: Information Retrieval, Reading Comprehension, TableQA and Question Generation |
| 📓 [Tutorials: Jupyter Notebooks](https://github.com/primeqa/primeqa/tree/main/notebooks) | Notebooks to get started on QA tasks |
| 💻 [Examples: Applying PrimeQA on various QA tasks](https://github.com/primeqa/primeqa/tree/main/examples) | Example scripts for fine-tuning PrimeQA models on a range of QA tasks |
| 🤗 [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing) | Upload and share your fine-tuned models with the community |
| ✅ [Pull Request](https://primeqa.github.io/primeqa/pull_request_template.html) | PrimeQA Pull Request |
| 📄 [Generate Documentation](https://primeqa.github.io/primeqa/README.html) | How Documentation works |        
| 🛠 [Orchestrator Service REST Microservice](https://primeqa.github.io/primeqa/orchestrator.html) | Proof-of-concept code for PrimeQA Orchestrator microservice |        
| 📖 [Tooling UI](https://primeqa.github.io/primeqa/tooling_ui.html) | Demo UI |        

## ❤️ PrimeQA collaborators include       

| | | | |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
|<img width="75" alt="stanford" src="docs/_static/img/collab-stanford-circle.png">| Stanford NLP |<img width="75" alt="i" src="docs/_static/img/collab-i-circle.png">| University of Illinois |
|<img width="75" alt="stuttgart" src="docs/_static/img/collab-stuttgart-circle.png">| University of Stuttgart | <img width="75" alt="notredame" src="docs/_static/img/collab-notredame-circle.png">| University of Notre Dame |
|<img width="75" alt="ohio" src="docs/_static/img/collab-ohio-circle.png">| Ohio State University |<img width="75" alt="carnegie" src="docs/_static/img/collab-carnegie-circle.png">| Carnegie Mellon University |
|<img width="75" alt="massachusetts" src="docs/_static/img/collab-massachusetts-circle.png">| University of Massachusetts |<img width="75" alt="ibm" src="docs/_static/img/collab-ibm.png">| IBM Research |
| | | | |


<br>
<br>
<br>
<br>
<div align="center">
    <img width="30" alt="primeqa" src="docs/_static/primeqa_logo.png">
</div>
