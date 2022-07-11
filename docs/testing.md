# Testing

## Unit Tests

Make sure the `tests` extras are installed before running the unit tests.

From there you can run the tests via pytest, for example:
```shell
pytest --cov primeqa --cov-config .coveragerc tests/
```

For more information, see:
- Our [tox.ini](https://github.ibm.com/ai-foundation/PrimeQA/blob/master/tox.ini)
- The [pytest](https://docs.pytest.org) and [tox](https://tox.wiki/en/latest/) documentation

### Continuous Integration

PrimeQA uses GitHub Actions to ensure all unit tests pass before a PR can be merged.
Further, multiple versions of Python are tested in parallel.
Additionally, a **minimum test coverage of 60%** overall is targeted.

Coverage is measured on the `primeqa` package and does not include `examples`.

PrimeQA CI uses GitHub hosted runners.  Details of the hardware resources can be found here: https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources

PrimeQA tests require more than the maximum 7GB RAM available on GitHub hosted runners.  To workaround this, the tests are launched as multiple sequential processes.  This also requires adjusting the coverage requirements to account for the subset of tests that are going to be run. 

If the job exits with `RC=137`, this is most likely a memory issue, you may have to update the GitHub Actions [workflow file](../.github/workflows/linux-ci.yml) and launch the test that were added as a separate step.

Note: the CI sometimes fails even when there are no test errors due to issues in downloading resources during the test.  
This is a [known issue](https://zenhub.ibm.com/workspaces/primeqa-61eed731a578f53e48934109/issues/ai-foundation/primeqa/82).
As mentioned in the issue the workaround is to re-run the failing test in Travis.
This will typically manifest as one of the Python versions failing (e.g. 3.8) and others
passing (e.g. 3.7 and 3.9).
