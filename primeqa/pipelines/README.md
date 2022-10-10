<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.pipelines

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# Pipelines

## Reader Components

### Extractive Reader

The Extractive Reader takes a question and a set of passages and returns an answer by extracting a span of text in the passages.
Follow the steps below to use the extractive reader:

- Step 1:  Initialize the reader. You can choose any of the MRC models we currently have [here](https://huggingface.co/PrimeQA).
```python
import json
from primeqa.pipelines.components.reader.extractive import ExtractiveReader
reader = ExtractiveReader("PrimeQA/tydiqa-primary-task-xlm-roberta-large")
```
- Step 2: Execute the reader in inference mode:
```python
question = ["Which country is Canberra located in?"]
context = ["""Canberra is the capital city of Australia. 
Founded following the federation of the colonies of Australia 
as the seat of government for the new nation, it is Australia's 
largest inland city"""]
answers = reader.apply(question,context)  
print(json.dumps(answers, indent=4))  
```