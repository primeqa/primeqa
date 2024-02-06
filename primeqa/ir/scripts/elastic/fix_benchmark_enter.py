import argparse
import json
import csv

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", help="output file path")
parser.add_argument("file", help="Input file")
args = parser.parse_args()

with open(args.file) as inp, open(args.output, "w") as out:
    reader = csv.reader(inp)
    writer = csv.writer(out)
    for row in reader:
        for index, entry in enumerate(row):
            if entry.find("\n")>=0:
                row[index] = entry.replace("\n", "\\n")
        writer.writerow(row)
