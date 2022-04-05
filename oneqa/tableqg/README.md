# Question Generation for TableQA
Generates synthetic question-answer pairs for tables. We first sample SQL queries from a given table, and then use a text-to-text transformer (T5) to transcribe the SQL query to a natural language question. For more details on the method check out our EMNLP 2021 paper [here](https://arxiv.org/abs/2109.07377). We also provide a flask API for this module.

Users can also use the question suggestion module to get next word suggestions (this is useful while typing the question). This suggestions come from a faiss index created from already generated questions.

## Pipeline
<img src="https://github.ibm.com/ai-4-interaction/QG-tableQA/blob/feature/cleanup/qg_pipeline.png" width="500" class="center">

## Setup
A conda env needs to be setup
```
conda create --prefix qg-env python=3.6
conda activate qg-env

pip install -r requirements.txt
```

If to run inference download the pre-trained T5 model folder from [here](https://ibm.box.com/s/d0bci7r75zydqtzedsqujxgkg0kby7oy) and place it inside "/oneqa/tableqg/tableQG/models/"

## Usage

### To run it from CMD line

```
  python ./oneqa/tableqg/OneQGDriver.py [Arg: JSON object describing the table]
  Exxample: python ./oneqa/tableqg/OneQGDriver.py '[{"header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School Team"],"rows": [["Antonio Lang", 21, "United States", "Guard-Forward", "1999-2000", "Duke"],["Voshon Lenard", 2, "United States", "Guard", "2002-03", "Minnesota"],["Martin Lewis", 32, "United States", "Guard-Forward", "1996-97", "Butler CC (KS)"],["Brad Lohaus", 33, "United States", "Forward-Center", "1996", "Iowa"],["Art Long", 42, "United States", "Forward-Center", "2002-03", "Cincinnati"]],"types":["text","real","text","text","text","text"]}]'

```

### To Run the Flask API Server 

```
  Python flask_app.py 

```

## To test the API

### To Generate question for a table 
```
curl -X POST http://0.0.0.0:5000/generate
   -H 'Content-Type: application/json'
   -d '{"num_samples":1,
 "table":[
    {"header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School Team"],
      "rows": [
            ["Antonio Lang", 21, "United States", "Guard-Forward", "1999-2000", "Duke"],
            ["Voshon Lenard", 2, "United States", "Guard", "2002-03", "Minnesota"],
            ["Martin Lewis", 32, "United States", "Guard-Forward", "1996-97", "Butler CC (KS)"],
            ["Brad Lohaus", 33, "United States", "Forward-Center", "1996", "Iowa"],
            ["Art Long", 42, "United States", "Forward-Center", "2002-03", "Cincinnati"]
        ],
     "types":["text","real","text","text","text","text"]
    }
    ]
}'
```

### To train a question generation model on your own data 
* Download training data from [here](https://ibm.box.com/s/0cm7k4eq7j9zg4vqo7rl529gwqwyuxj1)
* Place data inside the /oneqa/tableqg/tableQG/data folder
* run `sh train_t5_question_generation.sh`



