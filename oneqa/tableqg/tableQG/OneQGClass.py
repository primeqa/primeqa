import pwd
from tableQG.qg_inference import GenerateQuestions
from tqdm import tqdm
import kenlm

import time
from flask import Flask
from flask import request
from flask import jsonify, make_response, url_for
import json
import sys


class OneQG:
    qgtype = None 
    tableQG  = None
    oqg = None
    def __init__(self, typeOfQG):
        print ("in OneQG init")
        if typeOfQG == "Table":
            self.qgtype = typeOfQG
            print("to load TableQG model")
        #     gq = GenerateQuestions(
        # './tableQG/models/t5_model/')
            gq = GenerateQuestions('./oneqa/tableqg/tableQG/models/t5_model/')
            self.oqg = gq   
            self.tableQG=gq
        elif typeOfQG == "passage":
             print("passage QG is not supported yet")
             # instantiate passageQG and  assign it to oqg
        else : 
            print ("type of QG mentioned as "+ typeOfQG + ", it is not supported")
    

    def generate(self, contextObj):
        print ("in OneQG generate with type "+str(self.qgtype))
        if self.qgtype == "Table":
            if isinstance(self.oqg,GenerateQuestions):
                print ("to call TableQG.generate")
               # oqg=GenerateQuestions(self.oqg)
               # Qs =  self.oqg.generate_question(contextObj)
                Qs =  self.tableQG.generate_question(contextObj)
                print(Qs)

        





