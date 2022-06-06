# OneQA Documentation

OneQA uses Sphinx for documentation.  Before continuing make sure you have installed OneQA
with `docs` extras.  For example, from the top level of the project:

```shell
pip install .[docs]
```

The documentation can then be built with:

```shell
cd docs
make html
```

This will generate the documentation in HTML.

## Docker

The documentation can also be built and served from a Docker container.

```shell
VERSION=$(cat VERSION)
cd docs

# build image
./buildDocker.sh

# run container
docker run -p 80:80 --rm -d --name oneqa-docs oneqa-docs:${VERSION}
```