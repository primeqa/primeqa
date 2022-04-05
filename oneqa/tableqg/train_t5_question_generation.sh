mkdir -p tableQG/data/
mkdir -p tableQG/data/t5_model/
python tableQG/t5_generation.py train_data_path ./data/train_qg_t5.csv

