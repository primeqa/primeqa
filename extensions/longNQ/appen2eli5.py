import pandas as pd
from datetime import datetime
import json
import glob

name = {47200615:"Hee Dong", 46545976: "Eva Maria", 46092070: "Sara", 45676624: "Mohamed", 46373812: "Chie", 46545946: "Joekie", 46954475: "Arafat", 46994197: "Salim"}

'''
ELI5 Format
"id": "1oy5tc",
  "input": "in football whats the point of wasting the first two plays with a rush - up the middle - not regular rush plays i get those",
    "output": [
    {
      "answer": "Keep the defense honest, get a feel for the pass rush, open up the passing game. An offense that's too one dimensional will fail. And those rushes up the middle can be busted wide open sometimes for big yardage.",
      "meta": {
        "score": 3
      }
    },
    ]
     "passages": [
    {
      "pid": "387892::[4,4)",
      "title": "Rush (gridiron football)",
      "text": "Rushing, on offense, is running with the ball when starting from behind the line of scrimmage with an intent of gaining yardage. While this usually means a running play, any offensive play that 
does not involve a forward pass is a rush - also called a run. It is usually done by the running back after a handoff from the quarterback, although quarterbacks and wide receivers can also rush. The quarterba
ck will usually run when a passing play has broken down – such as when there is no receiver open to catch the ball – and there is room to",
      "score": 82.02098846435547
    },
]
'''

def process_data(data, valid_annotators=None, answer_field=""):

    eli5_formatted_data = {}
    answered = 0
    non_consecutive = 0
    selected = 0
    num_sentences = 0
    skip_count = 0
    bad_question = 0

    for _, row in data.iterrows():
        eli5_format = {}

        eli5_format['id'] = row['data']['question_id']
        eli5_format['input'] = row['data']['question']

        passage = {}

        if 'title' in row['data']:
            passage['title'] = row['data']['title']
        else:
            passage['title'] = None
        paragraph = ""   

        last_selected = -1
        is_nonconsecutive = False
        selected_sentences = []
        all_sentences = []
        i = 0
        num_sentences += len(row['data']['long_answer'])
        for sentence in row['data']['long_answer']:
            if 'paragraph_sentences' in row['results']['judgments'][0]['data'] and sentence in row['results']['judgments'][0]['data']['paragraph_sentences']:
                selected += 1
                if sentence.endswith("&"):
                    sentence = sentence[:-1]
                selected_sentences.append(sentence)
                if last_selected != -1 and last_selected != i-1:
                    is_nonconsecutive = True
                last_selected = i 
            i += 1
            all_sentences.append(sentence)
            paragraph += sentence + " "
        passage["text"] = paragraph
        eli5_format['passages'] = [passage]
        passage['sentences'] = all_sentences
        eli5_format['output'] = []

        is_answered = False

        for judgement in row['results']['judgments']:
            output = {}

            if valid_annotators is not None and name[judgement['worker_id']] not in valid_annotators:
                continue
            has_minimal_answer = False

            if 'minimal_text' in judgement['unit_data'] and judgement['unit_data']['minimal_text'] != '':
                has_minimal_answer = True
            
            if 'how_would_you_describe_the_questionanswer' not in judgement['data']:
                continue

            answer_type = judgement['data']['how_would_you_describe_the_questionanswer']

            output['answer'] = ""
            output['selected_sentences'] = selected_sentences
            output['meta'] = {"annotator": [judgement['worker_id']], "has_minimal_answer": has_minimal_answer, 'non_consecutive': is_nonconsecutive, "round": 1}
            
            if answer_type == 'complete' or answer_type == 'partial':
                # error
                if answer_field not in judgement['data']:
                    print(judgement)
                    continue
                # output['meta']['score'] = 3
                is_answered = True
                output['answer'] = judgement['data'][answer_field]
                eli5_format['output'].append(output)
                answered += 1
                if is_nonconsecutive:
                    non_consecutive += 1
            elif answer_type == "skip" or answer_type == "bad_paragraph":
                # output['meta']['score'] = 3
                output['answer'] = "NA"
                eli5_format['output'].append(output)
                skip_count += 1
                if is_nonconsecutive:
                    print("bad nonconsecutive -- skip/bad paragraph?") 
            else:
                if is_nonconsecutive:
                    print("bad nonconsecutive?") 
                bad_question+=1
        if len(eli5_format['output']) > 0: 
            eli5_formatted_data[eli5_format['id']] = eli5_format
    if answered != 0:
        print("answered: " + str(answered) + "/" + str(len(eli5_formatted_data)))
        print("skip: " + str(skip_count))
        print("bad: " + str(bad_question))
        print("non-consecutive: " + str(non_consecutive))
        print("selected: " + str(selected / len(eli5_formatted_data)))
        print("num sentences: " + str(num_sentences / len(eli5_formatted_data)))
    return eli5_formatted_data, answered, non_consecutive

def process_NA_data(data):

    eli5_formatted_data = {}
    answered = 0
    non_consecutive = 0
    selected = 0
    num_sentences = 0
    skip_count = 0
    bad_question = 0

    for _, row in data.iterrows():
        eli5_format = {}

        eli5_format['id'] = row['data']['id']
        eli5_format['input'] = row['data']['question']

        passage = {}
        passage['title'] = row['data']['title']
        passage["text"] = " ".join(row['data']['long_answer'])
        eli5_format['passages'] = [passage]
        eli5_format['output'] = []

        is_answered = False
        for judgement in row['results']['judgments']:
            output = {}
            
            answer_type = judgement['data']['how_would_you_describe_the_questionanswer']

            output['answer'] = ""
            output['meta'] = {"annotator": [judgement['worker_id'],row['data']['worker']], "has_minimal_answer": None, 'non_consecutive': None, "round": -1}
            
            if answer_type == 'complete' or answer_type == 'partial':
                answered += 1
            elif answer_type == "skip" or answer_type == "bad_paragraph" or answer_type == "bad_question":
                # output['meta']['score'] = 3
                output['answer'] = "NA"
                eli5_format['output'].append(output)
                skip_count += 1
            else:
                bad_question+=1
                print(answer_type)
        if len(eli5_format['output']) > 0: 
            eli5_formatted_data[eli5_format['id']] = eli5_format
    print("answered: " + str(answered))
    print("skip: " + str(skip_count))
    print("bad: " + str(bad_question))
    return eli5_formatted_data

def process_round2_data(eli5_formatted_data, data, answer_field=""):
    answered = 0
    num_sentences = 0

    for _, row in data.iterrows():
        eli5_format = {}

        if row['data']['id'] in eli5_formatted_data:
            eli5_format = eli5_formatted_data[row['data']['id']]
        else:
            eli5_format['id'] = row['data']['id']
            eli5_format['input'] = row['data']['question']

            passage = {}
            passage['title'] = row['data']['title']
            paragraph = ""   

            num_sentences += len(row['data']['long_answer'])
            all_sentences = []
            for sentence in row['data']['long_answer']:
                paragraph += sentence + " "
                all_sentences.append(sentence)
            passage["text"] = paragraph
            passage["sentences"] = all_sentences
            eli5_format['passages'] = [passage]
            eli5_format['output'] = []

        is_answered = False
        for judgement in row['results']['judgments']:
            output = {}

            has_minimal_answer = False

            if ('did_you_need_to_edit_the_answer' in judgement['data'] and 'skip' not in judgement['data']['did_you_need_to_edit_the_answer']) \
            or ('did_you_need_to_edit_the_answer_check_all_that_are_applicable' in judgement['data'] and 'skip' not in judgement['data']['did_you_need_to_edit_the_answer_check_all_that_are_applicable']) :
                is_answered = True
            else:
                updated_answer = "NA"

            if 'minimal_text' in judgement['unit_data'] and judgement['unit_data']['minimal_text'] != '':
                has_minimal_answer = True
                
            
            if 'update_the_answer_here_if_needed' in judgement['data']:
                if judgement['data']['paragraph_sentences'][0].endswith("&"):
                        judgement['data']['paragraph_sentences'] = [x[:-1] for x in judgement['data']['paragraph_sentences']]

                updated_answer = judgement['data']['update_the_answer_here_if_needed']
                output['answer'] = updated_answer
                output['selected_sentences'] = judgement['data']['paragraph_sentences']
                output['meta'] = {"annotator": [judgement['worker_id']], "has_minimal_answer": has_minimal_answer, "round": 2}
                # output['meta']['score'] = 3
                eli5_format['output'].append(output)
            elif is_answered:
                if eli5_format['id'] not in eli5_formatted_data:
                    eli5_formatted_data[eli5_format['id']] = {}
                    eli5_formatted_data[eli5_format['id']]['output']
                # add annotator to other answer
                if eli5_format['id'] in eli5_formatted_data and len(eli5_formatted_data[eli5_format['id']]['output']) > 1:
                    print("Multiple answers already")
                    print(eli5_format)
                # keep selected sentences of second annotator
                if 'paragraph_sentences' in judgement['data']:
                    if judgement['data']['paragraph_sentences'][0].endswith("&"):
                        judgement['data']['paragraph_sentences'] = [x[:-1] for x in judgement['data']['paragraph_sentences']]
                    eli5_formatted_data[eli5_format['id']]['output'][0]['selected_sentences'] = judgement['data']['paragraph_sentences']
                if judgement['worker_id'] not in eli5_formatted_data[eli5_format['id']]['output'][0]['meta']['annotator']:
                    eli5_formatted_data[eli5_format['id']]['output'][0]['meta']['annotator'].append(judgement['worker_id'])
                eli5_formatted_data[eli5_format['id']]['output'][0]['meta']['round'] = 2
            # skip
            else:
                output['answer'] = None
                output['selected_sentences'] = None
                output['meta'] = {"annotator": [judgement['worker_id']], "has_minimal_answer": has_minimal_answer, "round": 2, "skip": True}
                eli5_format['output'].append(output)

        if is_answered:
            answered += 1
        eli5_formatted_data[eli5_format['id']] = eli5_format
    return eli5_formatted_data, answered

def main():

    run_na = False
    # these are the round 1 files
    file_names = glob.glob("/dccstor/srosent2/generative/appen/round1_jobs/*json.zip") #["/dccstor/srosent2/generative/appen/round1_jobs/job_2022794.json", "/dccstor/srosent2/generative/appen/round1_jobs/job_2035917.json", 
    # "/dccstor/srosent2/generative/appen/round1_jobs/job_2006984.json", "/dccstor/srosent2/generative/appen/round1_jobs/job_2004889.json", "/dccstor/srosent2/generative/appen/round1_jobs/job_2084633.json"]
    # these are the round 2 files. 
    #round2_files = glob.glob("/dccstor/srosent2/generative/appen/round2_jobs*/*json.zip")
    round2_files = glob.glob("/dccstor/srosent2/generative/appen/round2_jobs*/output/*json.zip")
    # these are the NA files.
    if run_na:
        na_files = "/dccstor/srosent2/generative/appen/no_answer_round2/annotated/*"
        file_names = glob.glob(na_files)

    # file_names = ["/dccstor/srosent2/generative/appen/round1_jobs/job_2111155.json"]
    # round2_files_test = glob.glob("/dccstor/srosent2/generative/appen/round2_jobs3/output/*")

    # If an example has multiple annotations keep it as dev set.    
    fid_data = {}

    answered = 0
    non_consecutive = 0
    i = 0
    # NA
    if run_na:
        answer_fields = ["type_your_answer_here_it_should_be_concise_and_only_come_from_the_passagetitle_"]
        valid_annotators = [None]
        output_file = "/dccstor/srosent2/generative/appen/NQ_formatted_NA.json"
        
        for input_file in file_names:
            print(input_file)
            data = pd.read_json(input_file, lines=True) 
            formatted_data = process_NA_data(data)
            fid_data.update(formatted_data)
            i += 1
    else:
        answer_field = "type_your_answer_here_it_should_be_concise_and_only_come_from_the_passagetitle_"
        valid_annotators = None
        output_file = "/dccstor/srosent2/generative/appen/NQ_formatted_answered_single-11.16.23.json"

        for input_file in file_names:
            
            if "1999101" in input_file or "1988758" in input_file:
                continue
            if "2004889" in input_file:
                answer_field = "type_your_answer_here_keep_your_answer_as_close_to_the_passage_as_possible_"
                valid_annotators = ["Arafat","Sara","Salim"]
            else:
                answer_field = "type_your_answer_here_it_should_be_concise_and_only_come_from_the_passagetitle_"
                valid_annotators = None
            print(input_file)
            data = pd.read_json(input_file, lines=True)
            formatted_data, answered_i, non_consecutive_i = process_data(data, valid_annotators=valid_annotators, answer_field=answer_field)
            fid_data.update(formatted_data)
            answered += answered_i
            non_consecutive += non_consecutive_i
            i += 1
        print("answered: " + str(answered) + "/" + str(len(fid_data)))
        print("non-consective: " + str(non_consecutive))

        for input_file in round2_files:
            print(input_file)
            round2data = pd.read_json(input_file, lines=True)
            fid_data, answered_i = process_round2_data(fid_data, round2data)

        two_annotator_data = []
        with open(output_file,'wb') as writer:
            for data in fid_data:
                if fid_data[data]['output'][0]["meta"]["non_consecutive"]:
                    # len(fid_data[data]['output']) > 1 or \
                    # len(fid_data[data]['output'][0]["meta"]["annotator"]) > 1 or \
                    
                    two_annotator_data.append(fid_data[data])
                    continue
                writer.write((json.dumps(fid_data[data]) + "\n").encode())

        with open("/dccstor/srosent2/generative/appen/NQ_formatted_answered_multiple-11.16.23.json",'wb') as writer:
            for data in two_annotator_data:
                writer.write((json.dumps(data) + "\n").encode())

    with open(output_file,'wb') as writer:
        for data in fid_data:
            writer.write((json.dumps(fid_data[data]) + "\n").encode())

    

if __name__ == '__main__':
    main()