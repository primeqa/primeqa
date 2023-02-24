### PrimeQA support training and inference of question answering models for table and text.

#### To train model on HybridQA dataset
```shell
python primeqa/mitqa/run_mitqa.py primeqa/mitqa/config/train_hybridqa.json
```

#### To train model on Open Domin OTTQA dataset
```shell
python primeqa/mitqa/run_mitqa.py primeqa/mitqa/config/train_ottqa.json
```

#### To do inference on the pretrained model on HybridQA dataset
```shell
python primeqa/mitqa/run_mitqa.py primeqa/mitqa/config/inference_hybridqa.json
```

#### To do inference on the pretrained model on OTTQA dataset
```shell
python primeqa/mitqa/run_mitqa.py primeqa/mitqa/config/inference_ottqa.json
```

#### Dataset and Performance
- Model trained on HybridQA dataset achieves: 65 % EM and 72 % F1 on dev set.
- Model trained on OTTQA dataset achievs:  36% EM and 41% F1 on dev set using top 10 tables.