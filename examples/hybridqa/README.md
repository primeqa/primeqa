## Training and infernce of hybridqa dataset

### Training on hybridQA dataset
- Create data root folder named `data/hybridqa`
- Get data from this (https://github.com/wenhuchen/HybridQA/tree/master/released_data) repo and copy to the data root folder.
- Create sub folders for row retriever and answer extractor models as `data/hybridqa/models/row_retriever` and `data/hybridqa/models/answer_extractor`
- Clone the this (https://github.com/wenhuchen/WikiTables-WithLinks) repositiory inside `data/hybridqa` to get the tables with the links to passages.
- After creating the folder structure and getting the raw data and linked tables create/check and update the training config file available at `primeqa/hybridq/config/train_hybridqa.json`.
- After checking/updating the config file run the following command to train the model on HybridQA dataset:
`python primeqa/hybridqa/run_hybridqa.py <config_file_path>`, this command will train a model on HybridQA dataset and store models inside these (`data/hybridqa/models/row_retriever` and `data/hybridqa/models/answer_extractor`) directories. This command will also run evaluation on the dev set and report the final prediction accuracy on the dev set.

### Training on OTTQA dataset

- Create data root folder named `data/ottqa`.
- Get data from this (https://github.com/wenhuchen/OTT-QA/tree/master/released_data) repo and copy to the data root folder.
- Create sub folders for row retriever and answer extractor models as `data/ottqa/models/table_retriever`, `data/ottqa/models/link_generator`, `data/ottqa/models/row_retriever` and `data/ottqa/models/answer_extractor`.
- Get passages and tables corpuses using the following commands:
`wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_plain_tables.json`
`wget https://opendomainhybridqa.s3-us-west-2.amazonaws.com/all_passages.json`
- Make sure tables and passages corpuses are in data root folder.
- Make sure preprocessed triples for training and evaluation of the table retrievers are in data root folder.
#### To train the table retriever model for OTTQA dataset
- Make sure to set the flag `train_tr` in the config file `primeqa/hybridqa/config/train_ottqa.json` and also check and update all other hyperparameter values if required.
#### To train other modules for OTTQA dataset
- run the following command:
`python primeqa/hybridqa/run_hybridqa.py <ottqa_train_config_file>`




