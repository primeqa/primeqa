import watson_nlp
from watson_nlp.blocks.syntax import izumo
from watson_nlp.blocks.entity_mentions import BERT
from rouge_score import rouge_scorer
import pandas as pd
import random
from tqdm import tqdm
# Step 1: shuffle n mentions in answer from paragraph.
# Step 2: If there is only one mention, shuffle from another paragraph (build as going along)
# Step 3: If there are no mentions shuffle nouns.
# Alternative/Future approach: select n random mentions/nouns from other answers - build a list.


# Load the syntax model for English
model_path = watson_nlp.download('noun-phrases_rbr_en_stock', parent_dir='/dccstor/srosent2/watson_nlp') #syntax_izumo_en_stock')
np_model = watson_nlp.load(model_path)

model_path = watson_nlp.download('entity-mentions_bert_multi_stock', parent_dir='/dccstor/srosent2/watson_nlp') #'entity-mentions_rbr_en_stock') #syntax_izumo_en_stock')
# model_path = watson_nlp.download('entity-mentions_transformer_multilingual_slate.270m')
entity_model = watson_nlp.load(model_path)

syntax_model_en = watson_nlp.load(watson_nlp.download('syntax_izumo_en_stock', parent_dir='/dccstor/srosent2/watson_nlp'))

rouge = rouge_scorer.RougeScorer(rouge_types=['rougeLsum'], split_summaries=True)

split = "dev"
longnq_file = f"/dccstor/srosent2/generative/appen/final/longNQ/{split}/longNQ_{split}_answerable.jsonl"

longNQ = pd.read_json(longnq_file, lines=True, orient='records')

randomly_change = 3
nounphrase_count = 0

preference_data = {}

with tqdm(total=longNQ.shape[0]) as pbar:    
    for i, row in longNQ.iterrows():
        pbar.update(1)
        question = row['input']
        # print(f"Question\t {question}")
        passages = row['passages'][0]
        title = passages['title']
        text = passages['text']
        answer = row['output'][0]['answer']
        # print(f"Passage\t {passages['text']}")
        # print(f"Answer\t {answer}")
        
        syntax_analysis_text_en = syntax_model_en.run(text, parsers=('token',))
        mentions_text_prediction = entity_model.run(syntax_analysis_text_en)

        passage_mentions = {}

        for mention in mentions_text_prediction.mentions:
            # print(f"{mention.text}: {rouge.score(mention.text,question)['rougeLsum'][1]}")
            if rouge.score(mention.text,question)['rougeLsum'][1] > .5:
                continue
            if mention.type not in passage_mentions:
                passage_mentions[mention.type] = set()
            passage_mentions[mention.type].add(mention.text)

        syntax_analysis_en = syntax_model_en.run(answer, parsers=('token',))
        mentions_prediction = entity_model.run(syntax_analysis_en) #, parsers=('token', 'lemma', 'part_of_speech', 'noun_phrases'))

        answer_mentions = []

        # change nouns/adjectives in answer that are not in the question.
        # So the question words should still be there but the answer words won't be the same, and thus likely incorrect.
        for mention in mentions_prediction.mentions:
            # print(f"{mention.text}: {rouge.score(mention.text,question)['rougeLsum'][1]}")
            if rouge.score(mention.text,question)['rougeLsum'][1] > .5:
                continue
            answer_mentions.append(mention)

        # get nouns if not enough mentions
        if len(answer_mentions) <= 1:
            nounphrase_count +=1
            np_predictions = np_model.run(text)

            for np_prediction in np_predictions.noun_phrases:
                if rouge.score(np_prediction.text,question)['rougeLsum'][1] > .5:
                    continue
                if "noun_phrases" not in passage_mentions:
                    passage_mentions["noun_phrases"] = set()
                passage_mentions["noun_phrases"].add(np_prediction.text)

            np_answer_predictions = np_model.run(answer)
            for np_prediction in np_answer_predictions.noun_phrases:
                # np_prediction.type = 'noun_phrases'
                if rouge.score(np_prediction.text,question)['rougeLsum'][1] > .5:
                    continue
                answer_mentions.append(np_prediction)

        # print(f"passage mentions\t {passage_mentions.keys()}")
        # print(f"answer mentions\t {len(answer_mentions)}")

        n_mentions_changed = 0

        answer_mention_tochange = []
        answer_mention_replacements = []

        # the mentions must be updated in reverse to avoid changing the offsets
            # don't shuffle randomly pick while iterating backwards.
        adversarial_answer = answer
        for answer_mention in reversed(answer_mentions):
            if answer_mention.text in answer:

                try:
                    answer_mention_type = answer_mention.type 
                except: # no type, then its a noun phrase
                    answer_mention_type = 'noun_phrases'

                if random.random() > min(1,randomly_change/len(answer_mentions)):
                    continue

                if  answer_mention_type in passage_mentions and len(passage_mentions[answer_mention_type]) > 1:
                    replacement_mention = random.choice(list(passage_mentions[answer_mention_type]))
                    while replacement_mention == answer_mention.text:
                        replacement_mention = random.choice(list(passage_mentions[answer_mention_type]))
                else:
                    # no option to change (there is only 1 mention or its not in the passage)
                    if len(passage_mentions) == 1:
                        continue
                    random_type = random.choice(list(passage_mentions.items())) 
                    while random_type == answer_mention_type:
                        random_type = random.choice(list(passage_mentions.items()))[1] 
                    replacement_mention = random.choice(list(random_type[1]))
                # adversarial_answer = answer[:answer_mention.span.begin] + "|" + replacement_mention + "|" + answer[answer_mention.span.end:]
                adversarial_answer = adversarial_answer[:answer_mention.span.begin] + replacement_mention + adversarial_answer[answer_mention.span.end:]
        preference_data[row['id']] = {'id':row['id']}
        preference_data[row['id']]['chosen'] = f"{title}: {text}\nquestion: {question} answer:{answer}"
        preference_data[row['id']]['rejected'] = f"{title}: {text}\nquestion: {question} answer:{adversarial_answer}"
        if i > 10:
            break
        # print(f"adversarial answer\t {adversarial_answer}")
        # print("-------")
    
pd.DataFrame.from_dict(preference_data, orient='index').to_csv(f"/dccstor/srosent3/long_nq/preference_data/faithful/{split}_answerable_sub10.csv", index=False)
print(f"{nounphrase_count}/{i} noun phrases")