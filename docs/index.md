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

![Build Status](https://github.com/primeqa/primeqa/actions/workflows/primeqa-ci.yml/badge.svg)
[![LICENSE|Apache2.0](https://img.shields.io/github/license/saltstack/salt?color=blue)](https://www.apache.org/licenses/LICENSE-2.0.txt)
[![sphinx-doc-build](https://github.com/primeqa/primeqa/actions/workflows/sphinx-doc-build.yml/badge.svg)](https://github.com/primeqa/primeqa/actions/workflows/sphinx-doc-build.yml)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fprimeqa.github.io%2Fprimeqa&count_bg=%23953DC8&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=Visitors&edge_flat=false)]()

PrimeQA is a public open source repository that enables researchers and developers to train state-of-the-art models for question answering (QA). By using PrimeQA, a researcher can replicate the experiments outlined in a paper published in the latest NLP conference while also enjoying the capability to download pre-trained models (from an online repository) and run them on their own custom data. PrimeQA is built on top of the [Transformers](https://github.com/huggingface/transformers) toolkit and uses [datasets](https://huggingface.co/datasets/viewer/) and [models](https://huggingface.co/PrimeQA) that are directly downloadable.


The models within PrimeQA supports End-to-end Question Answering. PrimeQA answers questions via 
- [Information Retrieval](https://github.com/primeqa/primeqa/tree/main/primeqa/ir): Retrieving documents and passages using both traditional (e.g. BM25) and neural (e.g. ColBERT) models
- [Multilingual Machine Reading Comprehension](https://huggingface.co/ibm/tydiqa-primary-task-xlm-roberta-large): Extract and/ or generate answers given the source document or passage.
- [Multilingual Question Generation](https://huggingface.co/PrimeQA/mt5-base-tydi-question-generator): Supports generation of questions for effective domain adaptation over [tables](https://huggingface.co/PrimeQA/t5-base-table-question-generator) and [multilingual text](https://huggingface.co/PrimeQA/mt5-base-tydi-question-generator).

Some examples of models (applicable on benchmark datasets) supported are :
- [Traditional IR with BM25](https://github.com/primeqa/primeqa/tree/main/primeqa/ir/) Pyserini
- [Neural IR with ColBERT](https://github.com/primeqa/primeqa/tree/main/primeqa/ir), DPR (coming soon): to replicate the experiments that [Dr. Decr](https://huggingface.co/ibm/DrDecr_XOR-TyDi_whitebox) (Li et. al, 2022) performed to reach the top of the XOR TyDI leaderboard. Collaboration with [Stanford NLP](https://nlp.stanford.edu/) IR led by [Chris Potts](https://web.stanford.edu/~cgpotts/) & [Matei Zaharia](https://cs.stanford.edu/~matei/).
- [Machine Reading Comprehension with XLM-R](https://github.com/primeqa/primeqa/tree/main/primeqa/mrc): to replicate the experiments to get to the top of the TyDI leaderboard similar to the performance of the IBM GAAMA system. Coming soon: code to replicate GAAMA's performance on Natural Questions. 
- [Multimedia QA over news & movies](https://arxiv.org/abs/2112.10728): coming soon! to replicate the experiments run over multi-hop QA over images, text over variety of domains. Collaboration with [UIUC Blender lab](https://blender.cs.illinois.edu/).

<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
# Getting Started

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} {fas}`check` Installation
:link: installation
:link-type: doc
New to PrimeQA? Check out the getting started guide.
+++
{fas}`arrow-right`
:::

:::{grid-item-card} {fas}`code` Development
:link: development
:link-type: doc
Saw a typo in the documentation? Want to improve
existing functionalities? The contributing guidelines will guide
you through the process of improving PrimeQA.
+++
{fas}`arrow-right`
:::

:::{grid-item-card} {fas}`vial` Testing
:link: testing
:link-type: doc
Our goal is that every module and package should have a thorough set of unit tests. 
These tests should exercise the full functionality as well as its robustness 
to erroneous or unexpected input argument.
+++
{fas}`arrow-right`
:::

::::

```{eval-rst}
.. toctree::
    :maxdepth: 2
    :hidden:

    installation
    development
    testing
    api/index
```
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please -->

# Learn more

| Section | Description |
|-|-|
| [Quick tour: Entry Points for PrimeQA](https://github.com/primeqa/primeqa/tree/main/primeqa) | Different entry points for PrimeQA: Information Retrieval, Reading Comprehension, TableQA and Question Generation |
| [Tutorials: Jupyter Notebooks](https://github.com/primeqa/primeqa/tree/main/notebooks) | Notebooks to get started on QA tasks |
| [Examples: Applying PrimeQA on various QA tasks](https://github.com/primeqa/primeqa/tree/main/examples) | Example scripts for fine-tuning PrimeQA models on a range of QA tasks |
| [Model sharing and uploading](https://huggingface.co/docs/transformers/model_sharing) | Upload and share your fine-tuned models with the community |
| [Pull Request](pull_request_template.md) | PrimeQA Pull Request |
| [Generate Documentation](README.md) | How Documentation works |       

## PrimeQA collaborators include       

````{grid} 1 1 2 3
```{grid-item-card}
:img-top: _static/img/collab-stanford.png
:class-item: collab-item
```

```{grid-item-card}
:img-top: _static/img/collab-i.png
:class-item: collab-item-large
```

```{grid-item-card}
:img-top: _static/img/collab-stuttgart-black.png
:class-item: collab-item-small
```

```{grid-item-card}
:img-top: _static/img/collab-ohio.png
:class-item: collab-item
```

```{grid-item-card}
:img-top: _static/img/collab-carnegie.png
:class-item: collab-item
```

```{grid-item-card}
:img-top: _static/img/collab-massachusetts.png
:class-item: collab-item
```

```{grid-item-card}
:img-top: _static/img/collab-notredame.png
:class-item: collab-item-small
```

```{grid-item-card}
:img-top: _static/img/collab-utdallas.png
:class-item: collab-item-small
```
````