from tqdm import tqdm

def assign_ids(raw_dataset):
    """
    The assign_ids function takes in a dataset and assigns unique ids to each question.
    It does this by keeping track of the previous question id, and if it is the same as the current one,
    it adds an underscore and a number to indicate that it is not unique. It also keeps track of how many times
    this happens for each particular question.
    
    Args:
        raw_dataset: Pass the raw dataset
    
    Returns:
        A list of dictionaries with the question_id modified to include a number after the question id
    """
    new_data =[]
    prev_qid = ""
    for d in tqdm(raw_dataset):
        if prev_qid ==d['question_id']:
            new_qid = d['question_id']+"_"+str(i)
            prev_qid = d['question_id']
            i+=1
        else:
            i=1
            new_qid = d['question_id']+"_"+str(0)
            prev_qid = d['question_id']
        d['question_id'] = new_qid
    return raw_dataset
    