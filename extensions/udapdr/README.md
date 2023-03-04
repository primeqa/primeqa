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

For running the end-to-end domain adaptation approach, please use the `scripts/DSP_Multiple_Reranker.py`

Here is an example command for running `scripts/DSP_Multiple_Reranker.py`:

## Citing

````
@misc{https://doi.org/10.48550/arxiv.2303.00807,
  doi = {10.48550/ARXIV.2303.00807},
  url = {https://arxiv.org/abs/2303.00807},
  author = {Saad-Falcon, Jon and Khattab, Omar and Santhanam, Keshav and Florian, Radu and Franz, Martin and Roukos, Salim and Sil, Avirup and Sultan, Md Arafat and Potts, Christopher},
  keywords = {Information Retrieval (cs.IR), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers},
  publisher = {arXiv},
  year = {2023}, 
  copyright = {Creative Commons Attribution 4.0 International}
}
````
