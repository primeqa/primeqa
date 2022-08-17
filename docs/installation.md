# Installation
[<i class="fas fa-edit"></i> Edit on GitHub](https://github.com/primeqa/primeqa/edit/main/docs/installation.md)        

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

Please note that dependencies (specified in [setup.py](https://github.com/primeqa/primeqa/blob/main/setup.py))
are pinned to provide a stable experience. When installing from source these can be modified, 
however this is not officially supported.

## JAVA requirements
Java 11 is required for BM25 retrieval.

Download Java 11 package from [https://jdk.java.net/archive/](https://jdk.java.net/archive/) and uncompress

Set **JAVA_HOME**:
```shell
export JAVA_HOME=<jdk-dir>
export PATH=$JAVA_HOME/bin:$PATH
```