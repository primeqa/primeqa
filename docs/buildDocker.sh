#!/usr/bin/env bash
set -xeuo pipefail

if [[ "$#" -ne 0 ]]; then
    echo "Usage: ./buildDocker.sh"
    exit 1
fi

GIT_REPO_ROOT=$(git rev-parse --show-toplevel)
pushd "${GIT_REPO_ROOT}"
trap popd EXIT

VERSION=$(cat VERSION)

docker build -f docs/Dockerfile -t "primeqa-docs:${VERSION}" .
