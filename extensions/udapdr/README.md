# UDAPDR

## Overview

This repository includes the code and datasets for the experiments in [UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers](https://arxiv.org/abs/2303.00807).

## Installation

Run the following commands to setup a conda environment:

````
conda create --name udapdr --file requirements.txt
conda activate udapdr
````

## Data

We include three sample synthetic query datasets for LoTTE, BEIR, and NQ. To generate the datasets, we used our corpus-adapted prompting approach from [the paper](https://arxiv.org/abs/2303.00807), leveraging both GPT-3 and Flan-T5 XXL.

- LoTTE Pooled Synthetic Queries
- BEIR Scifact Synthetic Queries
- NaturalQuestions Synthetic Queries

## Training and Inference
