import json
def read_data(file_name):
    """
    The read_data function reads in a json file and returns the data as a dictionary.
    
    
    Args:
        file_name: Specify the name of the file that we want to read
    
    Returns:
        A dictionary of data
    """
    with open(file_name, 'r') as fin:
        data = json.load(fin)
    return data