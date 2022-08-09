# Question Generation for TableQA
Generates synthetic question-answer pairs for tables. We first sample SQL queries from a given table, and then use a text-to-text transformer (T5) to transcribe the SQL query to a natural language question. For more details on the method check out our EMNLP 2021 paper [here](https://arxiv.org/abs/2109.07377).
Currently we support tableqg model training over wikisql data and passageqg model training over squad, squad_v2, tydiqa data. 

To use our pre-trained QG models with only few lines of code check out the inference notebooks here: ../notebooks/qg/tableqg_inference.ipynb, ./notebooks/qg/pasageqg_inference.ipynb. 


To train the QG models on the supported datasets check out the inference notebooks here: ../notebooks/qg/tableqg_training.ipynb, ./notebooks/qg/pasageqg_training.ipynb. 


## Pipeline
<img src="../../docs/_static/img/tableqg_pipeline.png" width="500" class="center">