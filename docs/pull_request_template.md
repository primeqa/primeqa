# PrimeQA Pull Request

## What does this PR do?

Closes #(issue)

Notes:

- Replace `(issue)` **above** ↑↑↑ with the issue this PR closes to automatically link the two.
This must be done when the PR is created.
- Add multiple `Closes #(issue)` as needed.
- If this PR is work towards but does not close an issue, simply tag the issue without mentioning `Closes`.

## Description

**Describe** the changes proposed by this PR below to give the reviewer context below ↓↓↓

(description)

### Request Review

Be sure to **request a review** from one or more reviewers (unless the PR is to an unprotected branch).

## Versioning

When opening a PR to make changes to PrimeQA (i.e. `primeqa/`) master, be sure to increment the version following
[semantic versioning](https://semver.org/).  The VERSION is stored [here](https://github.com/primeqa/primeqa/blob/main/VERSION) 
and is incremented using `bump2version {patch,minor,major}` as described in the development guide documentation (https://github.com/primeqa/primeqa/blob/main/docs/development.md).

 - [ ] Have you updated the VERSION?
 - [ ] Or does this PR not change the `primeqa` package or was not into master?

After pulling in changes from master to an existing PR, ensure the VERSION is updated appropriately.
This may require bumping the version again if it has been previously bumped.

If you're not quite ready yet to post a PR for review, feel free to open a [draft PR](https://github.blog/2019-02-14-introducing-draft-pull-requests/).

## Releases

### After Merging
If merging into master and VERSION was updated, after this PR is merged:
- [ ] [Create a release](https://docs.github.com/en/github/administering-a-repository/releasing-projects-on-github/managing-releases-in-a-repository#creating-a-release) from the master with version equal to [VERSION](https://github.com/primeqa/primeqa/blob/main/VERSION)
- [ ] Not merging to master or VERSION not updated

## Checklist

Review the following and mark as completed:
- [ ] [Tag an issue or issues](#what-does-this-pr-do) this PR addresses.
- [ ] Added [description](#description) of changes proposed.
- [ ] Review requested as [appropriate](#request-review).
- [ ] Version bumped as [appropriate](#Versioning).
- [ ] New classes, methods, and functions documented.
- [ ] Documentation for modified code is updated.
- [ ] Built documentation to confirm it renders as expected (see https://github.com/primeqa/primeqa/blob/main/docs/development.md).
- [ ] Code cleaned up and commented out code removed.
- [ ] Tests added to ensure all functionalities tested at >= 60% unit test coverage (see https://github.com/primeqa/primeqa/blob/main/docs/development.md).
- [ ] Code cleaned up and commented out code removed.
- [ ] Release created [as needed](#after-merging) after merging.