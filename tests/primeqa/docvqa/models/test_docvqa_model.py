import os
import pytest
import requests
from primeqa.docvqa.models.docvqa_model import DocVQAModel

@pytest.mark.parametrize("model_name",["impira/layoutlm-document-qa"])
def test_qg_model(model_name):
    docvqa_model = DocVQAModel(model_name)


    url = "https://templates.invoicehome.com/invoice-template-us-neat-750px.png"
    resp = requests.get(url, stream=True)
    with open("sample_image.png", "wb") as png:
        png.write(resp.content)

    image = "sample_image.png"
    queries = ["What is the invoice number?", "What is the due date mentioned?"]
    samples = [(image, queries)]
    
    output_answers = docvqa_model.predict(samples, page=1)

    if os.path.exists(image):
        os.remove(image)

    assert output_answers[0][queries[0]] == "us-001"
    assert output_answers[0][queries[-1]] == "26/02/2019"
