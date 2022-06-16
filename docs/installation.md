# Installation

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

Please note that dependencies (specified in [setup.py](https://github.ibm.com/ai-foundation/PrimeQA/blob/master/setup.py))
are pinned to provide a stable experience. When installing from source these can be modified, 
however this is not officially supported. See the development guide for more on dependency management.