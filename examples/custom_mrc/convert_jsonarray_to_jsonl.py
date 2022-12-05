import json
import jsonlines
import argparse
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def handle_args():
    usage = "Convert jsonarray to jsonlines"
    parser = argparse.ArgumentParser(usage)
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="File containing feedback data from the PrimeQA application",
    )
    parser.add_argument(
        "--output_file", type=str, required=True, help="output jsonl file"
    )
    args = parser.parse_args()
    logger.info(vars(args))
    return args

def convert(infile,outfile):
    json_array = json.load(open(infile))

    with open(outfile, 'wb') as f:
        writer = jsonlines.Writer(f)
        writer.write_all(json_array)
        writer.close()
        
if __name__=='__main__':
    args = handle_args()
    convert(args.input_file, args.output_file)
    logger.info(f"Wrote {args.output_file}")
    
