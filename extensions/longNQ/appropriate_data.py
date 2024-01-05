# use annotated data, mrc data, and full passage as preference data of "chosen" and "rejected"
import pandas as pd

split = "test"
longnq_file = f"/dccstor/srosent2/generative/appen/final/longNQ/{split}/longNQ_{split}_answerable.jsonl"

longNQ = pd.read_json(longnq_file, lines=True, orient='records', dtype={'id':str})

mrc_longnq_file = f"/dccstor/srosent2/primeqa/experiments/long_nq/{split}/output/eval_predictions.json"
mrcLongNQ = pd.read_json(mrc_longnq_file, orient='index')

preference_data_passage = {}
preference_data_mrc = {}

for i, row in mrcLongNQ.iterrows():
    longNQRow = longNQ[longNQ['id'] == row[0]['example_id']].iloc[0]
    id = longNQRow['id']
    answer = longNQRow['output'][0]['answer']
    text = longNQRow['passages'][0]['text']
    mrc_target = row[0]['span_answer_text']
    question = longNQRow['input']
    title = longNQRow['passages'][0]['title']
    preference_data_mrc[id] = {'id':id}
    preference_data_mrc[id]['chosen'] = f"{title}: {text}\nquestion: {question} answer:{answer}"
    preference_data_mrc[id]['rejected'] = f"{title}: {text}\nquestion: {question} answer:{mrc_target}"
    preference_data_passage[id] = {'id':id}
    preference_data_passage[id]['chosen'] = f"{title}: {text}\nquestion: {question} answer:{answer}"
    preference_data_passage[id]['rejected'] = f"{title}: {text}\nquestion: {question} answer:{text}"

pd.DataFrame.from_dict(preference_data_mrc, orient='index').to_csv(f"/dccstor/srosent3/long_nq/preference_data/appropriate_short/{split}_answerable.csv", index=False)
pd.DataFrame.from_dict(preference_data_passage, orient='index').to_csv(f"/dccstor/srosent3/long_nq/preference_data/appropriate_long/{split}_answerable.csv", index=False)