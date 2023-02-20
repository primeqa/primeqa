import requests
import time
import json
import logging
# adapted from: https://github.ibm.com/hendrik-strobelt/bloom_service

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
logger = logging.getLogger(__name__)

class LLMService:
    """
    This class provides connectivity to the BAM service
    """
    def __init__(self, token: str, base_url='https://bam-api.res.ibm.com/v0/generate', model_id="bigscience/bloom"):
        """_summary_

        Args:
            token (str): api key
            base_url (str, optional): Defaults to 'https://bam-api.res.ibm.com/v0/generate'.
            model_id (str, optional): Defaults to "bigscience/bloom".
        """
        self.token = token
        self.base_url = base_url
        self.model_id = model_id

    def generate(self, inputs: list,
                 max_new_tokens=3,
                 min_new_tokens=10,
                 temperature=0,
                 top_k=5,
                 top_p=1):
        """Call the BAM service to generate text

        Args:
            inputs (list): the context
            max_new_tokens (int, optional): min new tokens to generate. Defaults to 3.
            min_new_tokens (int, optional): max new tokens to generate. Defaults to 10.
            temperature (int, optional): Defaults to 0.
            top_k (int, optional): Defaults to 5.
            top_p (int, optional): Defaults to 1.

        Returns:
            _type_: generated data
        """

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
            logger.info(str(response.content))
            dict_error = json.loads(response.content)
            logger.info("Rate limited for: " + str(dict_error['extensions']['state']['expires_in_ms'] * .001) + " seconds")
            time.sleep(dict_error['extensions']['state']['expires_in_ms'] * .001)
        else:
            logger.info(str(response.content))
            return {"error": response.reason, "status": response.status_code}