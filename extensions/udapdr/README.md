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

We include three sample synthetic query datasets for LoTTE and BEIR. To generate the datasets, we used our corpus-adapted prompting approach from [the paper](https://arxiv.org/abs/2303.00807), leveraging both GPT-3 and Flan-T5 XXL.

- LoTTE Pooled Synthetic Queries
- BEIR Scifact Synthetic Queries

To download the question and documents sets for evaluation, please go to [the following download site](https://zenodo.org/record/7698919#.ZAOg5-zMKdY).

## Training and Inference

For running the end-to-end domain adaptation approach, please use the `scripts/DSP_Multiple_Reranker.py`

Here is an example command for running `scripts/DSP_Multiple_Reranker.py`:

````
python scripts/DSP_Multiple_Reranker.py \
       --chosen_LoTTE_split pooled \
       --chosen_LoTTE_type forum \
       --chosen_LoTTE_set dev \
       --LoTTE_or_BEIR LoTTE \
       --chosen_BEIR_set None \
       --chosen_BEIR_type None \
       --sample_count 20 \
       --reranker_count 5 \
       --devices ["cuda:0"] \
       --query_count 1000000 \
       --model_choice google/flan-t5-xxl \
       --gpt3_model_choice text-davinci-002 \
       --parallelization False \
       --dsp_prompting False \
       --use_FLAN_for_all_synthetic_query_generation False
````

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
