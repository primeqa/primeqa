import json
import sys
from tableQG.OneQGClass import OneQG

#example input = '[{"header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School Team"],"rows": [["Antonio Lang", 21, "United States", "Guard-Forward", "1999-2000", "Duke"],["Voshon Lenard", 2, "United States", "Guard", "2002-03", "Minnesota"],["Martin Lewis", 32, "United States", "Guard-Forward", "1996-97", "Butler CC (KS)"],["Brad Lohaus", 33, "United States", "Forward-Center", "1996", "Iowa"],["Art Long", 42, "United States", "Forward-Center", "2002-03", "Cincinnati"]],"types":["text","real","text","text","text","text"]}]'
if __name__ == "__main__":
   print("in main")
   inputjson = sys.argv[1]
   print("printing the system arg: ")
   print(inputjson)
   contextObj = json.loads(inputjson)
   oqg=  OneQG("Table")
   oqg.generate(contextObj)

