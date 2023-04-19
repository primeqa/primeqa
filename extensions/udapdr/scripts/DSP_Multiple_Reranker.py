
import os
import dsp
import torch
import openai

from dsp_utilities.generate_synthetic_queries import generate_synthetic_queries
from dsp_utilities.generate_ColBERTv2_zeroshot_results import generate_ColBERTv2_zeroshot_results
from dsp_utilities.train_reranker import train_reranker
from dsp_utilities.evaluate_reranker import evaluate_reranker
from dsp_utilities.generate_triples import generate_triples
from dsp_utilities.distill_triples_with_retriever import distill_triples_with_retriever
from dsp_utilities.evaluate_beir import evaluate_beir

from dsp_utilities.evaluate_lotte_rankings import evaluate_dataset

import pandas as pd
import numpy as np
from tqdm import tqdm
tqdm.pandas()

import random
from random import randrange
import re

import argparse

######################################################################

random_state = 43

np.random.seed(random_state)
random.seed(random_state)
torch.manual_seed(random_state)
os.environ['PYTHONHASHSEED'] = str(random_state)

######################################################################

def is_bad_question_problematic(given_good_question, given_bad_question):

    given_good_question = re.sub(r'[^\w\s]','', given_good_question).lower()
    given_bad_question = re.sub(r'[^\w\s]','', given_bad_question).lower()

    if given_good_question[:20] == given_bad_question[:20]:
    	return True 
    
    good_question_split = given_good_question.split(" ")
    bad_question_split = given_bad_question.split(" ")
    if len(list(set(good_question_split) & set(given_bad_question))) > (len(good_question_split) / 2):
    	return True 

    #############

    return False

######################################################################

# Use GPT-3 prompting to generate initial synthetic queries
def generate_synthetic_questions_with_GPT3(given_passage, prompt_number, given_good_question):

	given_passage = " ".join(given_passage.split(" ")[:192])

	if prompt_number == 0:
		given_prompt = "Write a Question answered by the given Passage.\n" #passage
		given_prompt += "Passage: " + given_passage + "\n" #" ".join(given_passage.split(" ")[:256])
		given_prompt += "Question:" #passage
	elif prompt_number == 1:
		given_prompt = "Example 1:\n"
		given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.\n"
		given_prompt += "Good Question: How much caffeine is ok for a pregnant woman to have?\n"
		given_prompt += "Bad Question: Is a little caffeine ok during pregnancy?\n\n"
		given_prompt += "Example 2:\n"
		given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
		given_prompt += "Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\n"
		given_prompt += "Bad Question: What fruit is native to Australia?\n\n"
		given_prompt += "Example 3:\n"
		given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.\n"
		given_prompt += "Good Question: Information on the Canadian Armed Forces size and history\n"
		given_prompt += "Bad Question: How large is the Canadian military?\n\n"
		given_prompt += "Example 4:\n"
		given_prompt += "Document: " + given_passage + "\n" # + "\n"
		given_prompt += "Good Question:"
	elif prompt_number == 2:
		given_prompt = "Example 1:\n"
		given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee\n"
		given_prompt += "Relevant Query: Is a little caffeine ok during pregnancy?\n"
		given_prompt += "Example 2:\n"
		given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
		given_prompt += "Relevant Query: What fruit is native to Australia?\n"
		given_prompt += "Example 3:\n"
		given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day\n"
		given_prompt += "Relevant Query: How large is the Canadian military?\n"
		given_prompt += "Example 4:\n"
		given_prompt += "Document: " + given_passage + "\n"
		given_prompt += "Relevant Query:"
	elif prompt_number == 3:
		given_prompt = "Retrieve a Query answered by the following Document.\n" #passage
		given_prompt += "Document: " + given_passage + "\n"
		given_prompt += "Query:"
	elif prompt_number == 4:
		given_prompt = "Design a Question that is answered by the following Passage.\n" #passage
		given_prompt += "Passage: " + given_passage + "\n"
		given_prompt += "Question:"
	elif prompt_number == -1:
		given_prompt = "Example 1:\n"
		given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.\n"
		given_prompt += "Good Question: How much caffeine is ok for a pregnant woman to have?\n"
		given_prompt += "Bad Question: Is a little caffeine ok during pregnancy?\n\n"
		given_prompt += "Example 2:\n"
		given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
		given_prompt += "Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\n"
		given_prompt += "Bad Question: What fruit is native to Australia?\n\n"
		given_prompt += "Example 3:\n"
		given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.\n"
		given_prompt += "Good Question: Information on the Canadian Armed Forces size and history\n"
		given_prompt += "Bad Question: How large is the Canadian military?\n\n"
		given_prompt += "Example 4:\n"
		given_prompt += "Document: " + given_passage + "\n" # + "\n"
		given_prompt += "Good Question: " + given_good_question + "\n"
		given_prompt += "Bad Question:"


	########################################################

	response = openai.Completion.create(model=gpt3_model_choice, prompt=given_prompt, temperature=0.0, max_tokens=32)
	query = response['choices'][0]['text']
	query = query.replace("\n","").replace("\t","")
	unprocessed_query = query

	########################################################

	if prompt_number == 1:
		
		if query.lower().find("bad question") != -1:
			bad_question_index = query.find("bad question")
			query = query[:bad_question_index]
		query = query.replace("good question", "").replace(": ","")
	
	elif prompt_number == -1:
		
		if query.lower().find("bad question") != -1:
			bad_question_index = query.find("bad question")
			query = query[bad_question_index:]
		query = query.replace("bad question", "").replace(": ","")

	if query.find("?") != -1:
		question_mark_index = query.find("?")
		query = query[:question_mark_index + 1]
	query = query.strip()

	#########################################################

	return query


########################################################

# Use DSP prompting to generate initial synthetic queries
def perform_synthetic_question_generation_with_dsp_prompting(given_passage, prompt_number, given_good_question):

    def vanilla_LM_QA(document: str) -> str:

        if prompt_number == 1 or prompt_number == 2:
            demos = dsp.sample(train, k=3) # k = number of examples to provide in prompt
            example = dsp.Example(document=document, demos=demos)
        else:
            example = dsp.Example(document=document, demos=[])

        example, completions = dsp.generate(qa_template)(example, stage='qa')

        if prompt_number == -1:
            return completions.good_question, completions.bad_question
        else:
            return completions.good_question

    ########################################################

    given_passage = " ".join(given_passage.split(" ")[:192])

    ########################################################

    if prompt_number == 0:

        Document = dsp.Type(prefix="Passage:", desc="${passage for generating the question}")
        Synthetic_Good_Question = dsp.Type(prefix="Question:", desc="${a question that is less than 20 words in length}", format=dsp.format_answers)

        qa_template = dsp.Template(instructions="Write a Question answered by the given Passage.", document=Document(), good_question=Synthetic_Good_Question())
        completed_good_question = vanilla_LM_QA(given_passage)

        return completed_good_question

    elif prompt_number == 1:

        train = [("We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.", ('How much caffeine is ok for a pregnant woman to have?'), ("Is a little caffeine ok during pregnancy?")),
                 ('Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.', ('What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?'), ("What fruit is native to Australia?")),
                 ("The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.", ("Information on the Canadian Armed Forces size and history"), ("How large is the Canadian military?")),]
        train = [dsp.Example(document=document, good_question=good_question, bad_question=bad_question) for document, good_question, bad_question in train]

        ########################################################

        Document = dsp.Type(prefix="Document:", desc="${document for generating the question}")
        Synthetic_Good_Question = dsp.Type(prefix="Good Question:", desc="${a good question that is less than 20 words in length}", format=dsp.format_answers)
        Synthetic_Bad_Question = dsp.Type(prefix="Bad Question:", desc="${a bad question that is less than 20 words in length}", format=dsp.format_answers)

        qa_template = dsp.Template(instructions="Write a good question that is answered by the given document.", document=Document(), good_question=Synthetic_Good_Question(), bad_question=Synthetic_Bad_Question())
        completed_good_question = vanilla_LM_QA(given_passage)

        return completed_good_question #, completed_bad_question

    elif prompt_number == 2:

        train = [("We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.", ("Is a little caffeine ok during pregnancy?")),
                 ('Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.', ("What fruit is native to Australia?")),
                 ("The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.", ("How large is the Canadian military?")),]
        train = [dsp.Example(document=document, good_question=good_question) for document, good_question in train]

        ########################################################

        Document = dsp.Type(prefix="Document:", desc="${document for generating the question}")
        Synthetic_Good_Question = dsp.Type(prefix="Relevant Query:", desc="${a relevant query that is less than 20 words in length}", format=dsp.format_answers)

        qa_template = dsp.Template(instructions="Write a relevant query that is answered by the given document.", document=Document(), good_question=Synthetic_Good_Question())
        completed_good_question = vanilla_LM_QA(given_passage)

        return completed_good_question

    elif prompt_number == 3:

        Document = dsp.Type(prefix="Document:", desc="${document for generating the question}")
        Synthetic_Good_Question = dsp.Type(prefix="Question:", desc="${a question that is less than 20 words in length}", format=dsp.format_answers)

        qa_template = dsp.Template(instructions="Retrieve a Query answered by the following Document.", document=Document(), good_question=Synthetic_Good_Question())
        completed_good_question = vanilla_LM_QA(given_passage)

        return completed_good_question

    elif prompt_number == 4:

        Document = dsp.Type(prefix="Passage:", desc="${passage for generating the question}")
        Synthetic_Good_Question = dsp.Type(prefix="Question:", desc="${a question that is less than 20 words in length}", format=dsp.format_answers)

        qa_template = dsp.Template(instructions="Design a Question that is answered by the following Passage.", document=Document(), good_question=Synthetic_Good_Question())
        completed_good_question = vanilla_LM_QA(given_passage)

        return completed_good_question

    elif prompt_number == -1:

        train = [("We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.", ('How much caffeine is ok for a pregnant woman to have?'), ("Is a little caffeine ok during pregnancy?")),
                 ('Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.', ('What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?'), ("What fruit is native to Australia?")),
                 ("The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.", ("Information on the Canadian Armed Forces size and history"), ("How large is the Canadian military?")),]
        train = [dsp.Example(document=document, good_question=good_question, bad_question=bad_question) for document, good_question, bad_question in train]

        ########################################################

        Document = dsp.Type(prefix="Document:", desc="${document for generating the question}")
        Synthetic_Good_Question = dsp.Type(prefix="Good Question:", desc="${a good question that is less than 20 words in length}", format=dsp.format_answers)
        Synthetic_Bad_Question = dsp.Type(prefix="Bad Question:", desc="${a bad question that is less than 20 words in length}", format=dsp.format_answers)

        qa_template = dsp.Template(instructions="Write a good question that is answered by the given document.", document=Document(), good_question=Synthetic_Good_Question(), bad_question=Synthetic_Bad_Question())
        completed_good_question, completed_bad_question = vanilla_LM_QA(given_passage)

        return completed_bad_question

########################################################

# Use FLAN prompting to generate initial synthetic queries
def generate_synthetic_questions_with_FLAN(given_passage, prompt_number, given_good_question):

	given_passage = " ".join(given_passage.split(" ")[:192])

	if prompt_number == 0:
		given_prompt = "Write a Question answered by the given Passage.\n" 
		given_prompt += "Passage: " + given_passage + "\n"
		given_prompt += "Question:" #passage
	elif prompt_number == 1:
		given_prompt = "Example 1:\n"
		given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.\n"
		given_prompt += "Good Question: How much caffeine is ok for a pregnant woman to have?\n"
		given_prompt += "Bad Question: Is a little caffeine ok during pregnancy?\n\n"
		given_prompt += "Example 2:\n"
		given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
		given_prompt += "Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\n"
		given_prompt += "Bad Question: What fruit is native to Australia?\n\n"
		given_prompt += "Example 3:\n"
		given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.\n"
		given_prompt += "Good Question: Information on the Canadian Armed Forces size and history\n"
		given_prompt += "Bad Question: How large is the Canadian military?\n\n"
		given_prompt += "Example 4:\n"
		given_prompt += "Document: " + given_passage 
	elif prompt_number == 2:
		given_prompt = "Example 1:\n"
		given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee\n"
		given_prompt += "Relevant Query: Is a little caffeine ok during pregnancy?\n"
		given_prompt += "Example 2:\n"
		given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
		given_prompt += "Relevant Query: What fruit is native to Australia?\n"
		given_prompt += "Example 3:\n"
		given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day\n"
		given_prompt += "Relevant Query: How large is the Canadian military?\n"
		given_prompt += "Example 4:\n"
		given_prompt += "Document: " + given_passage #+ "\n"
		given_prompt += "Relevant Query:"
	elif prompt_number == 3:
		given_prompt = "Retrieve a Query answered by the following Document.\n" #passage
		given_prompt += "Document: " + given_passage + "\n"
		given_prompt += "Query:"
	elif prompt_number == 4:
		given_prompt = "Design a Question that is answered by the following Passage.\n" #passage
		given_prompt += "Passage: " + given_passage + "\n"
		given_prompt += "Question:"
	elif prompt_number == -1:
		given_prompt = "Example 1:\n"
		given_prompt += "Document: We don't know a lot about the effects of caffeine during pregnancy on you and your baby. So it's best to limit the amount you get each day. If you are pregnant, limit caffeine to 200 milligrams each day. This is about the amount in 1½ 8-ounce cups of coffee or one 12-ounce cup of coffee.\n"
		given_prompt += "Good Question: How much caffeine is ok for a pregnant woman to have?\n"
		given_prompt += "Bad Question: Is a little caffeine ok during pregnancy?\n\n"
		given_prompt += "Example 2:\n"
		given_prompt += "Document: Passiflora herbertiana. A rare passion fruit native to Australia. Fruits are green-skinned, white fleshed, with an unknown edible rating. Some sources list the fruit as edible, sweet and tasty, while others list the fruits as being bitter and inedible.\n"
		given_prompt += "Good Question: What is Passiflora herbertiana (a rare passion fruit) and how does it taste like?\n"
		given_prompt += "Bad Question: What fruit is native to Australia?\n\n"
		given_prompt += "Example 3:\n"
		given_prompt += "Document: The Canadian Armed Forces. 1 The first large-scale Canadian peacekeeping mission started in Egypt on November 24, 1956. 2 There are approximately 65,000 Regular Force and 25,000 reservist members in the Canadian military. 3 In Canada, August 9 is designated as National Peacekeepers' Day.\n"
		given_prompt += "Good Question: Information on the Canadian Armed Forces size and history\n"
		given_prompt += "Bad Question: How large is the Canadian military?\n\n"
		given_prompt += "Example 4:\n"
		given_prompt += "Document: " + given_passage + "\n"
		given_prompt += "Good Question: " + given_good_question 


	########################################################

	input_ids = flan_tokenizer.encode(given_prompt, max_length=2048, truncation=True, return_tensors='pt').to(device)
	if input_ids.shape[0] != 1 or input_ids.shape[1] >= 2048:
		print(input_ids.shape)
		print(input_ids.shape[0])
		print(input_ids.shape[1])
		print("Major error! Sequence length exceeds max length")
		#assert False
		return ""
	outputs = flan_model.generate(
	    input_ids=input_ids,
	    max_length=32,
	    do_sample=True,
	    top_p=0.95,
	    num_return_sequences=1)

	query = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)

	########################################################

	if prompt_number != -1:
		if query.lower().find("bad question") != -1:
			bad_question_index = query.lower().find("bad question")
			query = query[:bad_question_index]
		query = query.replace("Good Question", "").replace(": ","")
		query = query.replace("Relevant Query", "").replace(": ","")
	else:
		if query.lower().find("bad question") != -1:
			bad_question_index = query.lower().find("bad question")
			query = query[bad_question_index:]
		query = query.replace("Bad Question", "").replace(": ","")
		query = query.replace("Good Question", "").replace(": ","")
		query = query.replace("Relevant Query", "").replace(": ","")


	########################################################

	return query

########################################################

def end_to_end_reranker_training(given_prompt, given_device, given_process_number, queue, synthetic_queries_filename, synthetic_qas_filename, zeroshot_ranking, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder):

	print("Starting end-to-end process!")

	reranker_checkpoint_path, reranker_results_filename, reranker_success_at_five, baseline_success_at_5 = train_reranker(zeroshot_ranking, synthetic_queries_filename, synthetic_qas_filename, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, given_device, given_process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder)

	print("Completed step #3!")

	reranker_performance, baseline_performance = evaluate_reranker(reranker_checkpoint_path, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, given_device, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder)
	
	print("reranker_performance: " + str(reranker_performance))
	print("baseline_performance: " + str(baseline_performance))

	print("Completed step #4!")

	triples_for_distillation = generate_triples(reranker_results_filename, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, given_process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder)

	print("Completed step #5! Adding results to queue")

	queue.put((triples_for_distillation, synthetic_queries_filename, given_prompt, given_device, given_process_number))

######################################################################

def combine_rerankers_and_evaluate(triples_and_queries_list):

	distilled_checkpoint = "DSP_Experiments/msmarco.psg.kldR2.nway64.ib__colbert-400000"
	for triples_and_queries in triples_and_queries_list:

		distilled_checkpoint = distill_triples_with_retriever(triples_and_queries[0], triples_and_queries[1], distilled_checkpoint, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder)

	########################################

	print("Completed step #6!")

	distilled_ranking = generate_ColBERTv2_zeroshot_results(None, distilled_checkpoint, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, -1, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder, re_index=True)

	print("Completed step #7!")

	if LoTTE_or_BEIR == "LoTTE":

		evaluate_dataset(chosen_LoTTE_type, chosen_LoTTE_split, chosen_LoTTE_set, 5, 
			             "../ColBERT_FM/downloads/lotte/" + chosen_LoTTE_split + "/" + chosen_LoTTE_set + "/qas." + chosen_LoTTE_type + ".jsonl", 
			             distilled_ranking)

		evaluate_dataset(chosen_LoTTE_type, chosen_LoTTE_split, chosen_LoTTE_set, 20, 
			             "../ColBERT_FM/downloads/lotte/" + chosen_LoTTE_split + "/" + chosen_LoTTE_set + "/qas." + chosen_LoTTE_type + ".jsonl", 
			             distilled_ranking)

		evaluate_dataset(chosen_LoTTE_type, chosen_LoTTE_split, chosen_LoTTE_set, 100, 
			             "../ColBERT_FM/downloads/lotte/" + chosen_LoTTE_split + "/" + chosen_LoTTE_set + "/qas." + chosen_LoTTE_type + ".jsonl", 
			             distilled_ranking)
	
	elif LoTTE_or_BEIR == "BEIR":

		evaluate_dataset(chosen_LoTTE_type, chosen_LoTTE_split, chosen_LoTTE_set, 5, 
			             "../ColBERT_FM/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/qas.jsonl", 
			             distilled_ranking)

		evaluate_dataset(chosen_LoTTE_type, chosen_LoTTE_split, chosen_LoTTE_set, 20, 
			             "../ColBERT_FM/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/qas.jsonl", 
			             distilled_ranking)

		evaluate_dataset(chosen_LoTTE_type, chosen_LoTTE_split, chosen_LoTTE_set, 100, 
			             "../ColBERT_FM/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/qas.jsonl", 
			             distilled_ranking)

		evaluate_beir(distilled_ranking, chosen_BEIR_set, chosen_BEIR_type, downloads_folder)

######################################################################





if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument("--chosen_LoTTE_split", type=str, required=True)
	parser.add_argument("--chosen_LoTTE_type", type=str, required=True)
	parser.add_argument("--chosen_LoTTE_set", type=str, required=True)

	parser.add_argument("--LoTTE_or_BEIR", type=str, required=True)
	parser.add_argument("--chosen_BEIR_set", type=str, default=None, required=True)
	parser.add_argument("--chosen_BEIR_type", type=str, default=None, required=True)

	parser.add_argument("--sample_count", type=int, required=True)
	parser.add_argument("--reranker_count", type=int, default=5, required=True)
	parser.add_argument("--devices", type=list, default=["cuda:0", "cuda:1", "cuda:2", "cuda:3", "cuda:4"], required=False)

	parser.add_argument("--query_count", type=int, required=True)
	parser.add_argument("--model_choice", type=str, required=True)
	parser.add_argument("--gpt3_model_choice", type=str, required=True)
	parser.add_argument("--parallelization", type=bool, required=True)
	parser.add_argument("--dsp_prompting", type=bool, required=True, help="Use DSP prompting if true, use direct GPT-3 API prompting if false")
	parser.add_argument("--use_FLAN_for_all_synthetic_query_generation", default=False, type=bool, required=False, help="Use FLAN for initial query generation if true, use GPT-3 if false")
	parser.add_argument("--downloads_folder", type=str, default="../downloads", required=True, help="Folder containing LoTTE and BEIR directories")

	args = parser.parse_args()

	######################################################################

	chosen_LoTTE_split = args.chosen_LoTTE_split
	chosen_LoTTE_type = args.chosen_LoTTE_type
	chosen_LoTTE_set = args.chosen_LoTTE_set

	LoTTE_or_BEIR = args.LoTTE_or_BEIR
	chosen_BEIR_set = args.chosen_BEIR_set
	chosen_BEIR_type = args.chosen_BEIR_type

	sample_count = args.sample_count
	number_of_prompts = 5
	prompts_to_use = [0, 1, 2, 3, 4]
	reranker_count = args.reranker_count
	devices = args.devices

	query_count = args.query_count
	model_choice = args.model_choice
	gpt3_model_choice = args.gpt3_model_choice
	parallelization = args.parallelization
	dsp_prompting = args.dsp_prompting
	use_FLAN_for_all_synthetic_query_generation = args.use_FLAN_for_all_synthetic_query_generation
	downloads_folder = args.downloads_folder

	######################################################################

	if use_FLAN_for_all_synthetic_query_generation:
		flan_tokenizer = AutoTokenizer.from_pretrained(model_choice, max_length=2048, truncation=True)
		flan_model = AutoModelForSeq2SeqLM.from_pretrained(model_choice)

		chosen_device = "cuda:0"

		gpu_count = 4

		heads_per_gpu = len(flan_model.encoder.block) // gpu_count                                                                                                             
		device_map = {                                                                                                                                                      
			gpu: list(                                                                                                                                                      
			        range(                                                                                                                                                      
			            0 + (gpu * heads_per_gpu),                                                                                                                              
			            (0 + (gpu * heads_per_gpu)) + heads_per_gpu,                                                                                                            
			        )                                                                                                                                                           
			   )                                                                                                                                                               
			for gpu in range(gpu_count)                                                                                                                                     
		}                                                                                                                                                                   
		flan_model.parallelize(device_map)

		device = chosen_device
		device = torch.device(device)

	######################################################################

	openai_key = os.getenv('OPENAI_API_KEY')  # or replace with your API key (optional)
	colbert_server = 'http://ec2-44-228-128-229.us-west-2.compute.amazonaws.com:8893/api/search'

	lm = dsp.GPT3(model=gpt3_model_choice, api_key=openai_key)
	rm = dsp.ColBERTv2(url=colbert_server)

	dsp.settings.configure(lm=lm, rm=rm)

	########################################################

	if LoTTE_or_BEIR == "LoTTE":
		collection = pd.read_csv(downloads_folder + "/lotte/" + chosen_LoTTE_split + "/" + chosen_LoTTE_set + "/collection.tsv", sep="\t", header=None)
	elif LoTTE_or_BEIR == "BEIR":
		collection = pd.read_csv(downloads_folder + "/beir_datasets/" + chosen_BEIR_set + "/" + chosen_BEIR_type + "/collection.tsv", sep="\t", header=None)

	collection.columns = ['pid', 'passage']
	collection['original_pid'] = collection['pid']
	collection.set_index('original_pid', inplace=True)
	collection.sort_values('original_pid')

	print("collection")
	print(collection.shape)
	print(collection.head())

	######################################################################

	sub_collection = collection.sample(sample_count)

	print("reduced sample")
	print(sub_collection.columns)
	print(sub_collection.shape)
	print(sub_collection.head())

	######################################################################

	# Generate queries for each passage, using each prompt once

	for i in range(0, number_of_prompts):
		if i in prompts_to_use:
			if dsp_prompting:
				sub_collection['Generated_Question#' + str(i)] = sub_collection.progress_apply(lambda row: perform_synthetic_question_generation_with_dsp_prompting(row['passage'], i, ""), axis=1)
			elif use_FLAN_for_all_synthetic_query_generation:
				sub_collection['Generated_Question#' + str(i)] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_FLAN(row['passage'], i, ""), axis=1)
			else:
				sub_collection['Generated_Question#' + str(i)] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_GPT3(row['passage'], i, ""), axis=1)

	if dsp_prompting:
		sub_collection['Generated_Question#' + str(1)] = sub_collection.progress_apply(lambda row: perform_synthetic_question_generation_with_dsp_prompting(row['passage'], 1, ""), axis=1)
		sub_collection['Bad_Question'] = sub_collection.progress_apply(lambda row: perform_synthetic_question_generation_with_dsp_prompting(row['passage'], -1, row['Generated_Question#1']), axis=1)
	elif use_FLAN_for_all_synthetic_query_generation:
		sub_collection['Generated_Question#' + str(1)] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_FLAN(row['passage'], 1, ""), axis=1)
		sub_collection['Bad_Question'] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_FLAN(row['passage'], -1, row['Generated_Question#1']), axis=1)
	else:
		sub_collection['Generated_Question#' + str(1)] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_GPT3(row['passage'], 1, ""), axis=1)
		sub_collection['Bad_Question'] = sub_collection.progress_apply(lambda row: generate_synthetic_questions_with_GPT3(row['passage'], -1, row['Generated_Question#1']), axis=1)


	######################################################################

	if use_FLAN_for_all_synthetic_query_generation:
		del flan_model
		torch.cuda.empty_cache()

	######################################################################

	unique_prompts = []
	used_question_selection = set()

	print("Creating prompts")

	for i in range(reranker_count):

		current_questions = []
		current_bad_questions = []
		current_passages = []
		
		for j in range(1, 4):
			
			random_row = randrange(len(sub_collection))
			if number_of_prompts == len(prompts_to_use):
				random_column = prompts_to_use[j - 1]
			else:
				random_column = randrange(number_of_prompts)

			while random_row in used_question_selection or random_column not in prompts_to_use or is_bad_question_problematic(sub_collection.iloc[random_row]['Generated_Question#' + str(random_column)], sub_collection.iloc[random_row]['Bad_Question']):
				
				random_row = randrange(len(sub_collection))
				if number_of_prompts == len(prompts_to_use):
					random_column = prompts_to_use[j - 1]
				else:
					random_column = randrange(number_of_prompts)

			used_question_selection.add(random_row)
			current_questions.append(sub_collection.iloc[random_row]['Generated_Question#' + str(random_column)])
			current_bad_questions.append(sub_collection.iloc[random_row]['Bad_Question'])
			current_passages.append(sub_collection.iloc[random_row]['passage'])

		######################################################################

		current_prompt = ""
		for j in range(1, 4):
			current_prompt += "Example " + str(j) + ":\n"
			given_passage = " ".join(current_passages[j - 1].split(" ")[:192]).replace("\n", "").replace("\t", "")
			current_prompt += "Document: " + given_passage + "\n"
			current_prompt += "Good Question: " + current_questions[j - 1] + "\n"
			current_prompt += "Bad Question: " + current_bad_questions[j - 1] + "\n\n"

		print(current_prompt)
		print("------------------------------------------")

		unique_prompts.append(current_prompt)

	######################################################################

	import torch.multiprocessing as mp
	context = mp.get_context('spawn')
	queue = context.Queue()
	total_processes = []

	total_triples_and_synth_queries_for_distillation = []

	######################################################

	synthetic_queries_filenames = []
	synthetic_qas_filenames = []
	zeroshot_rankings = []

	print("Beginning Synthetic Query Generation!")

	# Perform synth query generation process and index generation before evaluation and distillation
	for prompt, chosen_device, process_number in zip(unique_prompts, devices, range(len(unique_prompts))):

	    synthetic_queries_filename, synthetic_qas_filename = generate_synthetic_queries(prompt, model_choice, query_count, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, 
	                                                                                    chosen_device, process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, parallelization, downloads_folder)
	    
	    zeroshot_ranking = generate_ColBERTv2_zeroshot_results(synthetic_queries_filename, downloads_folder + "/msmarco.psg.kldR2.nway64.ib__colbert-400000", 
	    													   chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, process_number, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder, re_index=True)

	    synthetic_queries_filenames.append(synthetic_queries_filename)
	    synthetic_qas_filenames.append(synthetic_qas_filename)
	    zeroshot_rankings.append(zeroshot_ranking)

	######################################################

	for prompt, chosen_device, process_number, synthetic_queries_filename, synthetic_qas_filename, zeroshot_ranking in zip(unique_prompts, devices, range(len(unique_prompts)), synthetic_queries_filenames, synthetic_qas_filenames, zeroshot_rankings):

	    print("Starting on a prompt!")

	    process = context.Process(target=end_to_end_reranker_training, args=(prompt,chosen_device,process_number,queue, synthetic_queries_filename, synthetic_qas_filename, zeroshot_ranking, chosen_LoTTE_split, chosen_LoTTE_type, chosen_LoTTE_set, LoTTE_or_BEIR, chosen_BEIR_set, chosen_BEIR_type, downloads_folder))
	    total_processes.append(process)
	    process.start()

	for finished_process in total_processes:
		finished_process.join()
		print("Joined a process!")

	print("Finished joining processes!")
	while not queue.empty():
		total_triples_and_synth_queries_for_distillation.append(queue.get())

	print("Finished gathering triples and synthetic queries!")
	print(len(total_triples_and_synth_queries_for_distillation))

	##############################

	print("Preparing to distill triples into retriever")

	combine_rerankers_and_evaluate(total_triples_and_synth_queries_for_distillation)



