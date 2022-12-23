### PrimeQA support trainining and inference of question answering models for table and text.

#### To train model on hybridqa dataset
```shell
python primeqa/hybridqa/run_hybridqa.py primeqa/hybridqa/config/train_hybridqa.json
```

#### To train model on Open Domin ottqa dataset
```shell
python primeqa/hybridqa/run_hybridqa.py primeqa/hybridqa/config/train_ottqa.json
```

#### To do inference on the pretrained model on hybridqa dataset
```shell
python primeqa/hybridqa/run_hybridqa.py primeqa/hybridqa/config/inference_hybridqa.json
```

#### To do inference on the pretrained model on ottqa dataset
```shell
python primeqa/hybridqa/run_hybridqa.py primeqa/hybridqa/config/inference_ottqa.json
```