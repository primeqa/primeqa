<!-- START sphinx doc instructions - DO NOT MODIFY next code, please -->
<details>
<summary>API Reference</summary>    

```{eval-rst}

.. autosummary::
    :toctree: _autosummary
    :template: custom-module-template.rst
    :recursive:
   
    primeqa.docvqa

```
</details>          
<br>
<!-- END sphinx doc instructions - DO NOT MODIFY above code, please --> 

# Document Visual Quesiton Answering (DocVQA)

Before continuing below make sure you have PrimeQA [installed](https://primeqa.github.io/primeqa/installation.html).

## Inference Example Usage
The following shows how to use the DocVQA component within PrimeQA to extract an answer given a question and an image:

 - Step 1:  Initialize the DocVQA model. The pre-trained model is available in huggingface and is trained on DocvQA dataset.
```python
import os
from primeqa.docvqa.models.docvqa_model import DocVQAModel

model = DocVQAModel("impira/layoutlm-document-qa")
```
- Step 2: Load a sample document for inference:
```python
import requests
url = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
resp = requests.get(url, stream=True)
image = "sample_image.png"
with open(image, "wb") as png:
    png.write(resp.content)
```
- Step 3: Execute the DocVQA model in inference mode:
```python
queries = ["What is the invoice number?", "What is the due date mentioned?"]
samples = [(image, queries)]
predictions = model.predict(samples, page=1)
print(json.dumps(predictions, indent=4))

if os.path.exists(image):
    os.remove(image)
```
The above statements will generate an output in the form of a dictionary:
```shell
[
    {
        "What is the invoice number?": "us-001",
        "What is the due date mentioned?": "26/02/2019"
    }
]
```

## Evaluate
If you want to perform a fully functional train and inference procedure for the MRC components, then the primary script to use is [run_mrc.py](https://github.com/primeqa/primeqa/blob/main/primeqa/mrc/run_mrc.py).  This runs a transformer-based MRC pipeline.

### Supported Datasets
Currently only [Docvqa](https://rrc.cvc.uab.es/?ch=17&com=downloads) dataset is supported. A custom dataset for evaluation is also supported with a directory structure similar to docvqa dataset.

### Example Usage

Only evaluation is supported:
```shell
python primeqa/docvqa/run_docvqa.py --dev_data_path ${DEV_DATA_PATH} --do_eval
```
This yields the following results:
```
***** eval metrics *****
ANLS Score:
```