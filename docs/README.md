# OneQA Documentation

OneQA uses Sphinx for documentation.  Before continuing make sure you have installed OneQA
with `docs` extras.  For example, from the top level of the project:

```shell
pip install .[docs]
```

The documentation can then be built with:

```shell
cd docs
sphinx-build -b html ./ ./generated/
```

This will generate the documentation in HTML.