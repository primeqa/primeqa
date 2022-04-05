import pwd
from tableQG.qg_inference import GenerateQuestions
from tqdm import tqdm
from querycompletion.query_completion import generate
import kenlm

import time
from flask import Flask
from flask import request
from flask import jsonify, make_response, url_for
import json
import sys


class OneQG:

    def __init__(self, typeOfQG):
        print ("in OneQG init")
        if typeOfQG == "table":
            gq = GenerateQuestions(
        './tableQG/models/t5_model/')
        elif typeOfQG == "passage":
             print("passage QG is not supported yet")
        else : 
            print ("type of QG not specified or supported")
    
    def getQG(self):
        print ("in OneQG getQG")
        return gq 
    
    def generate(self):
        print ("in OneQG generate")




