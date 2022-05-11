![OneQA Logo](logo.png)

# OneQA
OneQA to rule them all

[![Build Status](https://travis.ibm.com/ai-foundation/OneQA.svg?token=XcbF7zxMKHD12hqZiBwc&branch=master)](https://travis.ibm.com/ai-foundation/OneQA)


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
```

## Unit Tests

To run the unit tests you first need to [install OneQA](#Installation).
Make sure to install with the `[tests]` or `[all]` extras from pip.

From there you can run the tests via pytest, for example:
```shell
pytest --cov oneqa --cov-config .coveragerc tests/
```

For more information, see:
- Our [tox.ini](./tox.ini)
- The [pytest](https://docs.pytest.org) and [tox](https://tox.wiki/en/latest/) documentation


Our logo is adapted from [this](https://commons.wikimedia.org/wiki/File:One_Ring_Blender_Render.png)
image from Wikipedia under the
[Creative Commons Attribution-Share Alike 4.0 International](https://en.wikipedia.org/wiki/en:Creative_Commons) 
license.  It is shared under the same license.