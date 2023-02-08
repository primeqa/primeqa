## Training and inference on hybridqa dataset

### Training on hybridQA dataset
- Create data root folder named `data/hybridqa`
- Get data from this (https://github.com/wenhuchen/HybridQA/tree/master/released_data) repo and copy to the data root folder.
- Create sub folders for row retriever and answer extractor models as `data/hybridqa/models/row_retriever` and `data/hybridqa/models/answer_extractor`
- Clone the this (https://github.com/wenhuchen/WikiTables-WithLinks) repository inside `data/hybridqa` to get the tables with the links to passages.
- After creating the folder structure and getting the raw data and linked tables create/check and update the training config file available at `primeqa/hybridq/config/train_hybridqa.json`.
- After checking/updating the config file run the following command to train the model on HybridQA dataset:
`python primeqa/mitqa/run_mitqa.py <config_file_path>`, this command will train a model on HybridQA dataset and store models inside these (`data/hybridqa/models/row_retriever` and `data/hybridqa/models/answer_extractor`) directories. This command will also run evaluation on the dev set and report the final prediction accuracy on the dev set.

#### Replicating leaderboard results for hybridqa
- Download trained `row retriever` model from [here](https://huggingface.co/PrimeQA/MITQA_hybridqa_row_retriever/resolve/main/row_retriever.bin) and copy it into the `data/hybridqa/models/row_retriever/`.
- Check the config file to make sure the `model_name_path_ae` hyperparameter is set correctly to `PrimeQA/MITQA_hybridqa_multi_answer_answer_extractor`
- Run `python primeqa/mitqa/run_mitqa.py <inference_config_file_path>`

### Training on OTTQA dataset

- Create data root folder named `data/ottqa`.
- Get data from this (https://github.com/wenhuchen/OTT-QA/tree/master/released_data) repo and copy to the data root folder.
- Create sub folders for row retriever and answer extractor models as `data/ottqa/models/table_retriever`, `data/ottqa/models/link_generator`, `data/ottqa/models/row_retriever` and `data/ottqa/models/answer_extractor`.
- Get passages and tables corpuses using the following commands:
`wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json`
`wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json`
- Make sure tables and passages corpuses are in data root folder.
- Make sure preprocessed triples for training and evaluation of the table retrievers are in data root folder.
#### Replicating leaderboard results for OTTQA
- Create two sub folders `qry_encoder` and `ctx_encoder` inside `data/ottqa/models/table_retriever`.
- Download trained `query encoder` model and config file from [here](https://huggingface.co/PrimeQA/MITQA_OTTQA_DPR_Table_Retriever_Query_Encoder/tree/main) and copy it into `data/ottqa/models/table_retriever/qry_encoder`.
- Download trained `context encoder` model and config file from [here](https://huggingface.co/PrimeQA/MITQA_OTTQA_DPR_Table_Retriever_Context_Encoder/tree/main) and copy it into `data/ottqa/models/table_retriever/ctx_encoder`.
- Download trained `link predictor` model from [here](https://huggingface.co/PrimeQA/MITQA_OTTQA_Link_Predictor/resolve/main/model-ep9.pt) and copy it into the path `data/ottqa/models/link_generator`
- Download trained `row retriever` model from [here](https://huggingface.co/PrimeQA/MITQA_hybridqa_row_retriever/resolve/main/row_retriever.bin) and copy it into `data/hybridqa/models/row_retriever/`.
- Check the config file to make sure the `model_name_path_ae` hyperparameter is set correctly to `PrimeQA/MITQA_hybridqa_multi_answer_answer_extractor`
- Run `python primeqa/mitqa/run_mitqa.py <ottqa_inference_config_file_path>`

#### To train the table retriever model for OTTQA dataset
- Make sure to set the flag `train_tr` in the config file `primeqa/mitqa/config/train_ottqa.json` and also check and update all other hyperparameter values if required.
#### To train other modules for OTTQA dataset
- run the following command:
`python primeqa/mitqa/run_mitqa.py <ottqa_train_config_file>`




