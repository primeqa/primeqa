# Development

## Python

OneQA uses Python 3.7+.  Previous versions (e.g. 3.6) are unsupported.

## Dependency Management

OneQA uses pip for dependency management. Dependencies are controlled through
the `_deps` dict in [setup.py](https://github.ibm.com/ai-foundation/OneQA/blob/master/setup.py). 
Each entry maps a dependency to what it is required for.  All entries are alphabetized.
For example, `install` is for dependencies required to install/run OneQA (i.e. `install_requires`) 
whereas `docs` is for dependencies which are only needed to build the documentation. 
All values other than `install` correspond to pip extras. 
Additionally, there is an `all` extra that will install all dependencies.

A description of the pip extras is as follows:
- `docs`: for building documentation
- `dev`: for OneQA software development
- `install`: for installing and running OneQA
- `tests`: for running unit tests

## Versioning

OneQA uses [Semantic Versioning](https://semver.org/). Version can be incremented using `bump2version`.
Make sure the `dev` extras are installed.

Note: These commands need to be run from the project root.

```shell
bump2version {patch,minor,major}
git push  # above command creates a commit so we need to push
```

This will update the version accordingly and commit the change. 
The version change accompanying new code should be included in the Pull Request (PR). 
After the PR is merged make sure to create a new release with a descriptive title and
a version matching the new [VERSION](https://github.ibm.com/ai-foundation/OneQA/blob/master/VERSION).

## Coding Style

### Naming Conventions

- Class names should normally use the CapWords convention.
- Function names should be lowercase, with words separated by underscores as necessary to improve readability (i.e. snake_case).
- Method Names and Instance Variables follows the function naming rules: lowercase with words separated by underscores as necessary to improve readability (i.e. snake_case).
- Use one leading underscore only for non-public methods and instance variables (e.g. _private_attr).
- Constants are usually defined on a module level and written in all capital letters with underscores separating words. 
  - Examples include MAX_OVERFLOW and TOTAL. (from [pep8 guidelines](https://peps.python.org/pep-0008/#class-names))

### Folder Structure

TODO pending [decision](https://zenhub.ibm.com/workspaces/oneqa-61eed731a578f53e48934109/issues/ai-foundation/oneqa/165).

### Design Conventions

TODO pending [decision](https://zenhub.ibm.com/workspaces/oneqa-61eed731a578f53e48934109/issues/ai-foundation/oneqa/165).

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

#### Docstrings
Docstrings are used to generate the API reference. 
OneQA Docstrings follow the Google convention. 
See [Sec. 3.8](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for more details.

If you are using PyCharm, this docstring style can be selected from: `File --> Settings --> Tools --> Python Integrated Tools`.

#### Documentation Pages
Documentation pages (such as this one) are generated from a collection of Markdown files located
in `docs`.