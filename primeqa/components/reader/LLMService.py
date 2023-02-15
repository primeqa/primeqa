import requests
import time
import json
# adapted from: https://github.ibm.com/hendrik-strobelt/bloom_service

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class LLMService:
    def __init__(self, token: str, base_url='https://bam-api.res.ibm.com/v0/generate', model_id="bigscience/bloom"):
        self.token = token
        self.base_url = base_url
        self.model_id = model_id

    def generate(self, inputs: list,
                 max_new_tokens=3,
                 min_new_tokens=10,
                 temperature=0,
                 top_k=5,
                 top_p=1):

        parameters = {
            'max_new_tokens':max_new_tokens,
            'temperature':temperature,
            'min_new_tokens':min_new_tokens,
            'top_k':top_k,
            'top_p':top_p
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        json_data = {
            'model_id': self.model_id,
            'inputs': inputs,
            "parameters": parameters
        }
        response = requests.post(self.base_url,
                                 headers=headers, json=json_data, verify=False)

        if response.status_code == 201 or response.status_code == 200:
            r = response.json()
            r["request"] = json_data
            return r
        elif response.status_code == 429:
            print(str(response.content))
            dict_error = json.loads(response.content)
            print("Rate limited for: " + str(dict_error['extensions']['state']['expires_in_ms'] * .001) + " seconds")
            time.sleep(dict_error['extensions']['state']['expires_in_ms'] * .001)
        else:
            print(str(response.content))
            return {"error": response.reason, "status": response.status_code}