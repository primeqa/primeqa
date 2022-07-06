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
    <p>The prime repository for state-of-the-art Multilingual and Multimedia Question Answering research and development.</p>
</h3>

[![Build Status](https://app.travis-ci.com/primeqa/primeqa.svg?branch=main)](https://app.travis-ci.com/primeqa/primeqa)
[![LICENSE|Apache2.0](https://img.shields.io/github/license/saltstack/salt?color=blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)

PrimeQA is a public open source repository that enables researchers and developers to train state-of-the-art models for question answering (QA). By using PrimeQA, a researcher can replicate the experiments outlined in a paper published in the latest NLP conference while also enjoying the capability to download pre-trained models (from an online repository) and run them on their own custom data. PrimeQA is built on top of the [Transformers](https://github.com/huggingface/transformers) toolkit and uses [datasets](https://huggingface.co/datasets/viewer/) and [models](https://huggingface.co/IBM) that are directly downloadable.


The models within PrimeQA supports End-to-end Question Answering. PrimeQA answers questions via 
- [Information Retrieval](https://github.com/primeqa/primeqa/blob/main/examples/ir/run_ir.py): Retrieving documents and passages using both traditional (e.g. BM25) and neural (e.g. ColBERT) models
- [Multilingual Machine Reading Comprehension](https://huggingface.co/ibm/tydiqa-primary-task-xlm-roberta-large): Extract and/ or generate answers given the source document or passage.
- [Multilingual Question Generation](https://huggingface.co/PrimeQA/mt5-base-tydi-question-generator): Supports generation of questions for effective domain adaptation over [tables](https://huggingface.co/PrimeQA/t5-base-table-question-generator) and [multilingual text](https://huggingface.co/PrimeQA/mt5-base-tydi-question-generator).

Some examples of models (applicable on benchmark datasets) supported are :
- [Traditional IR with BM25](https://github.com/primeqa/primeqa/blob/main/examples/ir/run_bm25_retrieval.py) Pyserini
- [Neural IR with ColBERT, DPR (coming soon)](https://github.com/primeqa/primeqa/tree/main/primeqa/ir/dense/): to replicate the experiments that [Dr. Decr](https://huggingface.co/ibm/DrDecr_XOR-TyDi_whitebox) (Li et. al, 2022) performed to reach the top of the XOR TyDI leaderboard. Collaboration with [Stanford NLP](https://nlp.stanford.edu/) IR led by [Chris Potts](https://web.stanford.edu/~cgpotts/) & [Matei Zaharia](https://cs.stanford.edu/~matei/).
- [Machine Reading Comprehension with XLM-R](https://github.com/primeqa/primeqa/tree/main/primeqa/mrc): to replicate the experiments to get to the top of the TyDI leaderboard similar to the performance of the IBM GAAMA system. Coming soon: code to replicate GAAMA's performance on Natural Questions. 
- [Multimedia QA over news & movies](https://arxiv.org/abs/2112.10728): coming soon! to replicate the experiments run over multi-hop QA over images, text over variety of domains. Collaboration with [UIUC Blender lab](https://blender.cs.illinois.edu/).



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

## Learn more

| Section | Description |
|-|-|
| [Documentation](https://github.com/primeqa/primeqa/) | TODO: Full API documentation and tutorials |
| [Jupyter Notebooks](https://github.com/primeqa/primeqa/tree/main/notebooks) | Notebooks to get started on QA tasks |
| [Quick tour: Fine-tuning/usage scripts](https://github.com/primeqa/primeqa/tree/main/examples) | Example scripts for fine-tuning models on a range of QA tasks |
| [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing) | Upload and share your fine-tuned models with the community |


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
