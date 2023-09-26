# convert NQ to Tydi:
import spacy
from glob import glob

import gzip
import json

def load_json_from_file(gt_file_patterns):
    data = []
    if gt_file_patterns.endswith('gz'):
        f = gzip.open(gt_file_patterns, 'rt', encoding='utf-8')
    else:
        f = open(gt_file_patterns, 'rt', encoding='utf-8')
    return f.readlines()

# spacy.cli.download('en_core_web_sm')

'''
{
  "id": 5739443045617036288,
  "input": "rhyme scheme of the poem ode on a grecian urn",
  "passages": [
    {
      "title": "Ode on a Grecian Urn",
      "text": "`` Ode on a Grecian Urn '' is organized into ten - line stanzas , beginning with an ABAB rhyme scheme and ending with a Miltonic sestet ( 1st and 5th stanzas CDEDCE , 2nd stanza CDECED , and 3rd and 4th stanzas CDECDE ) . The same overall pattern is used in `` Ode on Indolence '' , `` Ode on Melancholy '' , and `` Ode to a Nightingale '' ( though their sestet rhyme schemes vary ) , which makes the poems unified in structure as well as theme . The word `` ode '' itself is of Greek origin , meaning `` sung '' . While ode - writers from antiquity adhered to rigid patterns of strophe , antistrophe , and epode , the form by Keats 's time had undergone enough transformation that it represented a manner rather than a set method for writing a certain type of lyric poetry . Keats 's odes seek to find a `` classical balance '' between two extremes , and in the structure of `` Ode on a Grecian Urn '' , these extremes are the symmetrical structure of classical literature and the asymmetry of Romantic poetry . The use of the ABAB structure in the beginning lines of each stanza represents a clear example of structure found in classical literature , and the remaining six lines appear to break free of the traditional poetic styles of Greek and Roman odes . ",
      "sentences": [
        "`` Ode on a Grecian Urn '' is organized into ten - line stanzas , beginning with an ABAB rhyme scheme and ending with a Miltonic sestet ( 1st and 5th stanzas CDEDCE , 2nd stanza CDECED , and 3rd and 4th stanzas CDECDE ) .",
        "The same overall pattern is used in `` Ode on Indolence '' , `` Ode on Melancholy '' , and `` Ode to a Nightingale '' ( though their sestet rhyme schemes vary ) , which makes the poems unified in structure as well as theme .",
        "The word `` ode '' itself is of Greek origin , meaning `` sung '' .",
        "While ode - writers from antiquity adhered to rigid patterns of strophe , antistrophe , and epode , the form by Keats 's time had undergone enough transformation that it represented a manner rather than a set method for writing a certain type of lyric poetry .",
        "Keats 's odes seek to find a `` classical balance '' between two extremes , and in the structure of `` Ode on a Grecian Urn '' , these extremes are the symmetrical structure of classical literature and the asymmetry of Romantic poetry .",
        "The use of the ABAB structure in the beginning lines of each stanza represents a clear example of structure found in classical literature , and the remaining six lines appear to break free of the traditional poetic styles of Greek and Roman odes ."
      ]
    }
  ],
  "output": [
    {
      "answer": "\"Ode on a Grecian Urn\" is organized into ten-line stanzas, beginning with an ABAB rhyme scheme and ending with a Miltonic sestet (1st and 5th stanzas CDEDCE, 2nd stanza CDECED, and 3rd and 4th stanzas CDECDE). While ode-writers from antiquity adhered to rigid patterns of strophe, antistrophe, and epode, the form by Keats's time had undergone enough transformation that it represented a manner rather than a set method for writing a certain type of lyric poetry.  Keats's odes seek to find a \"classical balance\" between two extremes, and in the structure of \"Ode on a Grecian Urn\", these extremes are the symmetrical structure of classical literature and the asymmetry of Romantic poetry. The use of the ABAB structure in the beginning lines of each stanza represents a clear example of structure found in classical literature, and the remaining six lines appear to break free of the traditional poetic styles of Greek and Roman odes.",
      "selected_sentences": [
        "`` Ode on a Grecian Urn '' is organized into ten - line stanzas , beginning with an ABAB rhyme scheme and ending with a Miltonic sestet ( 1st and 5th stanzas CDEDCE , 2nd stanza CDECED , and 3rd and 4th stanzas CDECDE ) .",
        "While ode - writers from antiquity adhered to rigid patterns of strophe , antistrophe , and epode , the form by Keats 's time had undergone enough transformation that it represented a manner rather than a set method for writing a certain type of lyric poetry .",
        "Keats 's odes seek to find a `` classical balance '' between two extremes , and in the structure of `` Ode on a Grecian Urn '' , these extremes are the symmetrical structure of classical literature and the asymmetry of Romantic poetry .",
        "The use of the ABAB structure in the beginning lines of each stanza represents a clear example of structure found in classical literature , and the remaining six lines appear to break free of the traditional poetic styles of Greek and Roman odes ."
      ],
         "meta": {
        "annotator": [
          46373812
        ],
        "has_minimal_answer": false,
        "non_consecutive": true,
        "round": 1
      }
}

'''

nlp = spacy.load('en_core_web_sm')

def convert(example):

    # for dev set taking first one as answer (this may need to be different)
    annotation = example['annotations'][0]

    passage_offsets = example['passage_answer_candidates'][annotation['passage_answer']['candidate_index']]
    passage_text = example['document_plaintext'].encode('utf-8')[passage_offsets['plaintext_start_byte']:passage_offsets['plaintext_end_byte']].decode('utf-8')
    
    passage_sentences = ""
    sentences = nlp(passage_text)
    split_sentences = []
    for sentence in sentences.sents:
        split_sentences.append(sentence.text)

    eli5_format = {}
    eli5_format['id'] = example['example_id']
    eli5_format['input'] = example["question_text"]

    passage = {}
    passage['title'] = example["document_title"]
    passage["text"] = passage_text
    passage['sentences'] = split_sentences
    eli5_format['passages'] = [passage]
    eli5_format['output'] = [{'answer': None, 'meta':None}]

    return eli5_format

def main():

    files = glob("/dccstor/srosent2/generative/appen/final/longNQ_test_unanswerable_tydi.jsonl")

    eli5data = []

    for file in files:
        lines = load_json_from_file(file)

        with open(file.replace('tydi.','eli5.'),'wb') as writer:
            for line in lines:
                writer.write((json.dumps(convert(json.loads(line))) + "\n").encode())

if __name__ == '__main__':
    main()    