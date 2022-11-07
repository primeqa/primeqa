#!/bin/bash
# this scrips is executed when a PR is accepted & merged
# if a README file has changes, then it is copied to its similar docs/ file
# to keep same information between docs (web page) and repo readmes

echo "----------------------------------"
echo "copy-readme-to-docs.sh"
echo "this scrips is executed when a PR is accepted & merged"
echo "if a README file has changes, then it is copied to its similar docs/ file"
echo "to keep same information between docs (web page) and repo readmes"
echo "----------------------------------"
echo "Script executed from: ${PWD}"
echo "----------------------------------"
echo "Git config"
git config --global user.name "github-actions[bot]"
git config --global user.email "41898282+github-actions[bot]@users.noreply.github.com"
echo "----------------------------------"
echo "Copying files ..."
cp primeqa/boolqa/README.md docs/api/boolqa/index.md
cp primeqa/calibration/README.md docs/api/calibration/index.md
cp primeqa/distillation/README.md docs/api/distillation/index.md
cp primeqa/ir/README.md docs/api/ir/index.md
cp primeqa/mrc/README.md docs/api/mrc/index.md
cp primeqa/pipelines/README.md docs/api/pipelines/index.md
cp primeqa/qg/README.md docs/api/qg/index.md
cp primeqa/tableqa/README.md docs/api/tableqa/index.md
cp primeqa/util/README.md docs/api/util/index.md
echo "----------------------------------"
echo "Git add"
git add docs/api/boolqa/index.md
git add docs/api/calibration/index.md
git add docs/api/distillation/index.md
git add docs/api/ir/index.md
git add docs/api/mrc/index.md
git add docs/api/pipelines/index.md
git add docs/api/qg/index.md
git add docs/api/tableqa/index.md
git add docs/api/util/index.md
echo "----------------------------------"
echo "Git commit & push"
git commit -m "github-actions[update api readme files in docs]"
git push
echo "----------------------------------"