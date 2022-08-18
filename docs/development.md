(development)=
# Development
<!-- [<i class="fas fa-edit"></i> Edit on GitHub](https://github.com/primeqa/primeqa/edit/main/docs/development.md)         -->

## Python

PrimeQA uses Python 3.7+.  Previous versions (e.g. 3.6) are unsupported.

## Dependency Management

PrimeQA uses pip for dependency management. Dependencies are controlled through
the `_deps` dict in [setup.py](https://github.com/primeqa/primeqa/blob/main/setup.py). 
Each entry maps a dependency to what it is required for.  All entries are alphabetized.
For example, `install` is for dependencies required to install/run PrimeQA (i.e. `install_requires`) 
whereas `docs` is for dependencies which are only needed to build the documentation. 
All values other than `install` correspond to pip extras. 
Additionally, there is an `all` extra that will install all dependencies.

A description of the pip extras is as follows:
- `docs`: for building documentation
- `dev`: for PrimeQA software development
- `install`: for installing and running PrimeQA
- `tests`: for running unit tests

## Versioning

PrimeQA uses [Semantic Versioning](https://semver.org/). Version can be incremented using `bump2version`.
Make sure the `dev` extras are installed.

Note: These commands need to be run from the project root.

```shell
bump2version {patch,minor,major}
git push  # above command creates a commit so we need to push
```

This will update the version accordingly and commit the change. 
The version change accompanying new code should be included in the Pull Request (PR). 
After the PR is merged make sure to create a new release with a descriptive title and
a version matching the new [VERSION](https://github.com/primeqa/primeqa/blob/main/VERSION).

## Coding Style

### Naming Conventions

- Class names should normally use the CapWords convention.
- Function names should be lowercase, with words separated by underscores as necessary to improve readability (i.e. snake_case).
- Method Names and Instance Variables follows the function naming rules: lowercase with words separated by underscores as necessary to improve readability (i.e. snake_case).
- Use one leading underscore only for non-public methods and instance variables (e.g. _private_attr).
- Constants are usually defined on a module level and written in all capital letters with underscores separating words. 
  - Examples include MAX_OVERFLOW and TOTAL. (find more details from [pep8 guidelines](https://peps.python.org/pep-0008/#class-names))


### Folder Structure
- Any new contribution to PrimeQA should have its own folder under `/primeqa/<functional_folder>` where the folder name should be a meaningful presentation of the functionality being added. For example: `/primeqa/mrc/`  , `/primeqa/ir` and so on.
- Before adding a new folder, please check if the broad functionality already exists in PrimeQA and if so, it can come under that folder. e.g. dense retriever and sparse retriever both comes under `/primeqa/ir/`
- Inside each functional folder, the code should be organized across common folder structure in PrimeQA as mentioned below:
  - data_models: codes relating to I/O format for different `tasks/models`.
  - processors: should have all data pre/post processors needed to convert user I/O to model I/O
  - models: modular functions related to model usage across different tasks e.g. using the trainer and/or using the (pre-)trained models.
  - trainers: main trainer module to train the functionality.
  - metrics: evaluation scripts to add support for different datasets.
  - any other helper functions, utilities might come under a /utils folder inside `/primeqa/<functional_folder>/` . This is to be done only if the existing common folder structures (as stated above) do not match the purpose of the util functions. 
- [ToDo] In addition to the resspective functional folders, primeqa should have a common folder that hosts functionalities common across all the primeqa contributions such as the base methods in PrimeQA described in the next section.
- Note: PrimeQA has some task agnostic folders in its base such as 
  - `/docs`: contains the documentations (such as this one).
  - `/examples`: Sample commands to run each functionality. 
  - `/notebooks`: Example notebook demonstrating the usage of each functionality. 

### Base methods in PrimeQA:
- class **PrimeQATrainer(transformers.Trainer)**  # subclass HF Trainer
- train: **PrimeQATrainer.train()** method which can be extended by individual task trainers.
- evaluate: **PrimeQATrainer.evaluate()** method which can be extended by individual task evaluators. 
- Every new PrimeQA contribution (e.g. QG, Re2G, ListQA etc.) should subclass **PrimeQATrainer** for its trainer
- Common HF interfaces (likes of train(), evaluate() etc. ) should be used through **PrimeQA core classes** instead of directly from HF.  

### Design Conventions

Use the standard `logging` library over `print` calls.
Further, when creating classes whose methods need logging, create a 
`_logger` instance attribute and use that for logging.

```python
self._logger = logging.getLogger(self.__class__.__name__)  # create logger
self._logger.info("Log message")
```

Use the standard `logging` library over `print` calls.
Further, when creating classes whose methods need logging, create a 
`_logger` instance attribute and use that for logging.

```python
self._logger = logging.getLogger(self.__class__.__name__)  # create logger
self._logger.info("Log message")
```

### Documentation

Documentation is generated using Sphinx based on:

1. Docstrings
2. Documentation Pages

Make sure to have the `docs` extras installed before building the documentation.
The documentation can be built as shown below.

Note: These commands need to be run from the project root.

```shell
cd docs
sphinx-build -b html ./ ./generated/
```

#### Docstrings:
Docstrings are used to generate the API reference. 
PrimeQA Docstrings follow the Google convention. 
See [Sec. 3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for more details.
Such doctrings can be used to auto-generate code comments using Sphinx tool ([Example](https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html))

If you are using PyCharm, this docstring style can be selected from: `File --> Settings --> Tools --> Python Integrated Tools`.

#### Notebooks: 
Each contribution should have an easy to use notebook that exemplifiess how to instantiate and use the functionality. 
An example for MRC on Tydi is [here](https://github.com/primeqa/primeqa/blob/main/notebooks/mrc/tydiqa.ipynb)

#### Documentation Pages:
Documentation pages (such as this one) are generated from a collection of Markdown files located
in `docs`. It also contains a subfolder calles `_static/img` which contains any (legally usable) images needed for documentation purposes. 

#### [Deploying Sphinx Docs as project online](https://github.com/primeqa/primeqa/blob/main/docs/README.md#deploying-a-sphinx-project) 