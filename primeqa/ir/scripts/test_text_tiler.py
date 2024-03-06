import json

from text_tiler import TextTiler
# from primeqa.ir.scripts.old_elastic_ingestion import process_text
from primeqa.ir.scripts.elastic_ingestion import process_text
from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm

# text = ("...")  # omitted for brevity
parser = ArgumentParser()
parser.add_argument('--num_elements', type=int, default=100,
                    help="Number of text elements to test with (default 100)")
parser.add_argument('--max_length', type=int, required=True)
parser.add_argument('--stride', type=int, default=100)
parser.add_argument('--input', type=str, required=True,
                    help="The jsonl file with text elements to test.")
parser.add_argument("-v", "--verbose", action="store_true", help="Turns on verbose output.")
parser.add_argument("--debug_indices", nargs="+", type=int, default=None,
                    help="The offsets to debug")

args = parser.parse_args()
max_doc = args.max_length
stride = args.stride
verbose = args.verbose

title = "Data storage values - IBM Documentation"
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
tiler = TextTiler(max_doc_size=max_doc, stride=stride, tokenizer=tokenizer)

# Argument parser

num_elements = 0
num_failures = 0
failures = []
with open(args.input) as inp:
    if args.debug_indices is None:
        inds = range(0, args.num_elements)
    else:
        inds = args.debug_indices
    values = []
    for i, line in enumerate(inp):
        if i in inds:
            values.append(line)
        if i > inds[-1]:
            break

    for i, text in tqdm(zip(inds, values), desc="Processing documents: ", total=args.num_elements):
        data = json.loads(text)
        text_ = data['document']
        title_ = data['title']
        res1 = tiler.create_tiles(id_="id1", text=text_, title=title_,
                                  title_handling="all",
                                  template={'productId': 'documentation',
                                            'deliverableLoio': 'something',
                                            'filePath': 'something',
                                            'title': title_,
                                            'url': 'http://ibm.com/documentation/something/something.html',
                                            'app_name': ''
                                            })

        res2 = process_text(id="id1", text=text_, title=title_, max_doc_size=max_doc, stride=stride,
                            tiler=tiler,
                            tokenizer=tokenizer,
                            doc_url="http://ibm.com/documentation/something/something.html")

        if res1 != res2:
            num_failures += 1
            failures.append(i)
            if verbose:
                print("Test failed: {res1}\n{res2}".format(res1=res1, res2=res2))
                if len(res1) != len(res2):
                    print(f"The sequences have different lengths: {len(res1)} vs {len(res2)}:\n"
                          f" {res1}\n {res2}\n")


        num_elements += 1
        if num_elements > args.num_elements:
            break

print(f"Number of failures: {num_failures}/{args.num_elements}. Failed on {failures}")
