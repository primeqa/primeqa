import pandas as pd
from datetime import datetime
import json

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

    eli5_formatted_data = []
    answered = 0
    non_consecutive = 0
    selected = 0
    num_sentences = 0

    for _, row in data.iterrows():
        eli5_format = {}

        eli5_format['id'] = row['data']['question_id']
        eli5_format['input'] = row['data']['question']

        passage = {}
        passage['title'] = row['data']['title']
        paragraph = ""   

        last_selected = -1

        i = 0
        num_sentences += len(row['data']['long_answer'])
        for sentence in row['data']['long_answer']:
            if 'paragraph_sentences' in row['results']['judgments'][0]['data'] and sentence in row['results']['judgments'][0]['data']['paragraph_sentences']:
                selected += 1
                if last_selected != -1 and last_selected != i-1:
                    non_consecutive += 1
                last_selected = i 
            i += 1
            paragraph += sentence + " "
        passage["text"] = paragraph
        eli5_format['passages'] = [passage]
        eli5_format['output'] = []

        is_answered = False
        for judgement in row['results']['judgments']:
            output = {}

            if valid_annotators is not None and name[judgement['worker_id']] not in valid_annotators:
                continue
            has_minimal_answer = False

            if judgement['unit_data']['minimal_text'] != '':
                has_minimal_answer = True
            
            answer_type = judgement['data']['how_would_you_describe_the_questionanswer']

            output['answer'] = ""
            output['meta'] = {"score":0, "annotator": name[judgement['worker_id']], "has_minimal_answer": has_minimal_answer}
            
            if answer_type == 'complete' or answer_type == 'partial':
                # error
                if answer_field not in judgement['data']:
                    print(judgement)
                    continue
                output['meta']['score'] = 3
                is_answered = True
                output['answer'] = judgement['data'][answer_field]
                eli5_format['output'].append(output)
        if is_answered:
            answered += 1
            eli5_formatted_data.append(eli5_format)
    print("answered: " + str(answered) + "/" + str(len(eli5_formatted_data)))
    print("non-consective: " + str(non_consecutive))
    print("selected: " + str(selected / len(eli5_formatted_data)))
    print("num sentences: " + str(num_sentences / len(eli5_formatted_data)))
    return eli5_formatted_data, answered, non_consecutive

def main():
    file_names = ["/dccstor/srosent2/generative/appen/job_2022794.json", "/dccstor/srosent2/generative/appen/job_2035917.json", 
    "/dccstor/srosent2/generative/appen/job_2006984.json", "/dccstor/srosent2/generative/appen/job_2004889.json"]
    answer_fields = ["type_your_answer_here_it_should_be_concise_and_only_come_from_the_passagetitle_", "type_your_answer_here_it_should_be_concise_and_only_come_from_the_passagetitle_",
        "type_your_answer_here_it_should_be_concise_and_only_come_from_the_passagetitle_","type_your_answer_here_keep_your_answer_as_close_to_the_passage_as_possible_"]
    valid_annotators = [None, None, ["Arafat","Sara","Salim"], ["Arafat","Sara","Salim"]]
    output_file = "/dccstor/srosent2/generative/appen/NQ_formatted_answered.json"

    fid_data = []

    answered = 0
    non_consecutive = 0
    i = 0
    for input_file in file_names:
        data = pd.read_json(input_file, lines=True)
        formatted_data, answered_i, non_consecutive_i = process_data(data, valid_annotators=valid_annotators[i], answer_field=answer_fields[i])
        fid_data.extend(formatted_data)
        answered += answered_i
        non_consecutive += non_consecutive_i
        i += 1
    print("answered: " + str(answered) + "/" + str(len(fid_data)))
    print("non-consective: " + str(non_consecutive))
    with open(output_file,'wb') as writer:
        for data in fid_data:
            writer.write((json.dumps(data) + "\n").encode())

    

if __name__ == '__main__':
    main()