<!---
Copyright 2022 PrimeQA team from IBM Research AI. All rights reserved.

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
    <p>The prime repository for state-of-the-art Question Answering research and development.</p>
</h3>



[![Build Status](https://travis.ibm.com/ai-foundation/PrimeQA.svg?token=XcbF7zxMKHD12hqZiBwc&branch=master)](https://travis.ibm.com/ai-foundation/PrimeQA)


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

