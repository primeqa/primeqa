# Testing

## Unit Tests

Make sure the `tests` extras are installed before running the unit tests.

From there you can run the tests via pytest, for example:
```shell
pytest --cov oneqa --cov-config .coveragerc tests/
```

For more information, see:
- Our [tox.ini](https://github.ibm.com/ai-foundation/OneQA/blob/master/tox.ini)
- The [pytest](https://docs.pytest.org) and [tox](https://tox.wiki/en/latest/) documentation

### Continuous Integration

OneQA uses Travis CI to ensure all unit tests pass before a PR can be merged.
Further, multiple versions of Python are tested in parallel.
Additionally, a minimum test coverage of 60% is enforced.

Note: the CI sometimes fails even when there are no test errors due to issues in downloading
resources during the test.  This is a [known issue](https://zenhub.ibm.com/workspaces/oneqa-61eed731a578f53e48934109/issues/ai-foundation/oneqa/82).
As mentioned in the issue the workaround is to re-run the failing test in Travis.
This will typically manifest as one of the Python versions failing (e.g. 3.8) and others
passing (e.g. 3.7 and 3.9).