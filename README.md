# OneQA
OneQA to rule them all

[![Build Status](https://travis.ibm.com/IBM-Research-AI/OneQA.svg?token=XcbF7zxMKHD12hqZiBwc&branch=master)](https://travis.ibm.com/IBM-Research-AI/OneQA)


## Getting Started

## Installation

```shell
# cd to project root

# If you want to run on GPU make sure to install torch appropriately

# E.g. for torch + CUDA 11.3:
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113

# Torch 1.8.2 LTS + CUDA 11.1:
pip install torch==1.8.2+cu111 -f https://download.pytorch.org/whl/lts/1.8/torch_lts.html

# Install as editable (-e) or non-editable using pip, with extras (e.g. tests) as desired
# Example installation commands:

# Minimal install (non-editable)
pip install .

# Full install (editable)
pip install -e .[all]
```
