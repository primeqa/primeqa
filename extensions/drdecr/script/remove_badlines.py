import argparse

def reader(file_name): 
	for row in open(file_name, 'r'):
		yield row

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file", type=str, required=True
    )
    parser.add_argument(
        "--output_file", type=str, required=True
    )
    args = parser.parse_args()
    row_gen = reader(args.input_file)
    with open(args.output_file, 'w') as f:
        for idx, row in enumerate(row_gen):
            if len(row.strip().split('\t')) == 2:
                f.write(row)

if __name__=='__main__':
	main()