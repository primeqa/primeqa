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

# PrimeQA
<h3 align="center">
    <p>The prime repository for state-of-the-art Multilingual Question Answering research and development.</p>
</h3>

[![Build Status](https://travis.ibm.com/ai-foundation/PrimeQA.svg?token=XcbF7zxMKHD12hqZiBwc&branch=master)](https://travis.ibm.com/ai-foundation/PrimeQA)
[![LICENSE|Apache2.0](https://img.shields.io/github/license/saltstack/salt?color=blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)
PrimeQA is a public open source repository that provides researchers and developers to train state-of-the-art models for question answering (QA). By using PrimeQA, a researcher would be able to replicate the experiments outlined in a paper published in a latest NLP conference while also enjoying the capability to download the trained models (from an online repository) and run them on their own custom data. 

The models within PrimeQA supports End-to-end Question Answering. These answering questions via 
- Information Retrieval: Retrieving documents and passages using both traditional (e.g. BM25) and neural (e.g. ColBERT) models
- Machine Reading Comprehension: Extract/ or generate answers given the source document or passage.
- Question Generation: Supports generation of questions for effective domain adaptation.

Some examples of models (on datasets) supported are :
- [Traditional IR with BM25] Pyserini
- [Neural IR with ColBERT, DPR (coming soon)]: PrimeQA will allow you to replicate the experiments that Dr. Decr (Li et. al, 2022) performed to reach the top of the XOR TyDI leaderboard.
- [Machine Reading Comprehension with XLM-R]: PrimeQA lets one replicate experiments to get to the top of the TyDI leaderboard similar to the performance of IBM GAAMA system. Coming soon: code to replicate GAAMA's performance on Natural Questions. 



## Getting Started

## Installation

```shell
# cd to project root

# If you want to run on GPU make sure to install torch appropriately

# E.g. for torch 1.11 + CUDA 11.3:
pip install 'torch~=1.11.0' --extra-index-url https://download.pytorch.org/whl/cu113

# Install as editable (-e) or non-editable using pip, with extras (e.g. tests) as desired
# Example installation commands:

# Minimal install (non-editable)
pip install .

# Full install (editable)
pip install -e .[all]
```

Please note that dependencies (specified in [setup.py](./setup.py)) are pinned to provide a stable experience.
When installing from source these can be modified, however this is not officially supported.

## JAVA requirements
Java 11 is required for BM25 retrieval. 

Download Java 11 package from https://jdk.java.net/archive/ and uncompress

Set JAVA_HOME:
```shell
export JAVA_HOME=<jdk-dir>
export PATH=$JAVA_HOME/bin:$PATH
```

## Unit Tests

To run the unit tests you first need to [install PrimeQA](#Installation).
Make sure to install with the `[tests]` or `[all]` extras from pip.

From there you can run the tests via pytest, for example:
```shell
pytest --cov PrimeQA --cov-config .coveragerc tests/
```

For more information, see:
- Our [tox.ini](./tox.ini)
- The [pytest](https://docs.pytest.org) and [tox](https://tox.wiki/en/latest/) documentation
