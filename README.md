# OneQA
OneQA to rule them all

[![Build Status](https://travis.ibm.com/IBM-Research-AI/OneQA.svg?token=XcbF7zxMKHD12hqZiBwc&branch=master)](https://travis.ibm.com/IBM-Research-AI/OneQA)


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