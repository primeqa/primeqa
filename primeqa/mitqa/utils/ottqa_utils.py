from tqdm import tqdm

def assign_ids(raw_dataset):
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
    