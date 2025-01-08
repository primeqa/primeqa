# asqa
from datasets import load_dataset
import pandas as pd
import tqdm 
from rouge_score import rouge_scorer
rouge = rouge_scorer.RougeScorer(rouge_types=['rouge1',], split_summaries=False)

datasets = load_dataset("din0s/asqa")
for split, dataset in datasets.items():
    seen_context = 0
    json_data = []

    for i in tqdm.tqdm(range(len(dataset))):
        json_example = {}
        # dataset conversion
        # query, input_urls, target
        json_example['input'] = dataset['ambiguous_question'][i]
        json_example['id'] = dataset['sample_id'][i]

        passages = []
        output = []

        seen_text = set()
        # max_len = -1

        for annotation in dataset['annotations'][i]:
            # keep the longest knowledge like in the paper oracle setup
            # if len(annotation['long_answer']) > max_len and len(annotation['knowledge']) > 0:
            for context in annotation['knowledge']:
                if context['content'] == None:
                    continue
                if context['content'].strip() in seen_text:
                    continue
                seen = False
                for text in seen_text:
                    rouge_score = rouge.score(text, context['content'])['rouge1'][2]

                    if rouge_score > .95:
                        seen = True
                        break
                if seen:
                    continue
                seen_text.add(context['content'].strip())
                passages.append({'title': context['wikipage'], 'text': context['content'].strip()})
            output.append({'answer': annotation['long_answer'], 'meta': {}})

        missing_contexts = False

        for qa_pair in dataset['qa_pairs'][i]:
            if qa_pair['context'] == "No context provided":
                missing_contexts = True
            elif qa_pair['context'].strip() not in seen_text:
                seen = False
                for text in seen_text:
                    rouge_score = rouge.score(text, qa_pair['context'])['rouge1'][2]

                    if rouge_score > .95:
                        seen_context += 1
                        seen = True
                        break
                if seen:
                    continue
                seen_text.add(qa_pair['context'].strip())
                passages.append({'title': qa_pair['wikipage'], 'text': qa_pair['context'].strip()})

        if len(passages) == 0 or len(output) == 0: # or missing_contexts:
            continue

        json_example['passages'] = passages
        json_example['output'] = output
        json_data.append(json_example)

    print(f"Seen context: {seen_context}")

    print(f"dump {split} {len(json_data)}")
    pd.DataFrame.from_dict(json_data).to_json(f"/dccstor/srosent2/generative/external_datasets/asqa/knowledge_and_context_nodups/{split}.jsonl", orient='records', lines=True)