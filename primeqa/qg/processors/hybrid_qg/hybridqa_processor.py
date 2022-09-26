import json
import numpy as np
from copy import deepcopy

import stanza
from tqdm import tqdm
from datasets import load_dataset
from primeqa.qg.utils.constants import QGSpecialTokens

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer


class HybridQADataset():
    def __init__(self, dataset_name='hybrid_qa'):
        """
        Class for sampling hybrid chains from a table and its linked passages, to be used for HybridQG training.
        Args:
            dataset_name (str): Supports only hybrid_qa for now.
        """
        self.dataset_name = dataset_name
        self.nlp_model = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)

        self.answer_separator = " %s " % (QGSpecialTokens.ans)
        self.hop_separator = " %s " % (QGSpecialTokens.hsep)
        self.meta_separator = " %s " % (QGSpecialTokens.header)
    
    def hybrid_chain_to_t5_sequence(self, qdict, chain_id=0):
        # puts hybrid chain in a format T5 can understand
        ans = qdict['answer_text']
        chain = qdict['hybrid_chain'][chain_id]['chain-text']
        meta = [qdict['title'], qdict['section_title']]
        chain_with_separators = self.hop_separator.join(chain)
        meta_with_separators = self.meta_separator.join([' ']+meta)
        source_text = "%s %s %s %s" % (ans, self.answer_separator, chain_with_separators.strip(), meta_with_separators.strip())	
        target_text = qdict['question']	
        return source_text, target_text

    def get_candidate_hybrid_chains(self, row, answer_node, num_hops=3, beam_size=5):
        """
            Extracts reasoning paths from a processed table row.
        Args:
            row (dict): A dictionary of a processed table row. 
            answer_node (str): Gold answer.
            num_hops (int): Number of hops to reach an answer from the question over the entity graph.
        """
        #row represents just one question with all the information necessary
        
        hybrid_chain_list = [{'chain':[answer_node], 'tfidf_mean':row[answer_node]['tfidf_score']}]
        for c in range(num_hops-1):
            # generating beam_size x beam_size hybrid chains
            new_hybrid_chain_list = []
            for i in range(len(hybrid_chain_list)):
                hchain = hybrid_chain_list[i]['chain']
                current_node = hchain[-1]
                tfidf_mean = hybrid_chain_list[i]['tfidf_mean']
    
                all_hops = row[current_node]['links']
    
                # we need to remove loops in the chain
                available_hops = [node for node in all_hops if node not in hchain]
    
                # Also not go back to another sentence of a passage already seen.
                passages_in_chain = ['_'.join(node.split('_')[:2]) for node in hchain if 'passage_' in node]
                nodes_to_remove = []
                for node in available_hops:
                    for psg_node in passages_in_chain:
                        if psg_node in node: # if 'passage_0' in 'passage_0_1'
                            nodes_to_remove.append(node)
                available_hops = [node for node in available_hops if node not in nodes_to_remove]
    
                tfidf_scores = [row[node]['tfidf_score'] for node in available_hops]
    
                high_value_node_ids = np.argsort(tfidf_scores)[::-1][:beam_size] # taking top 'beam_size' options
                for new_node in high_value_node_ids:
                    new_hchain = {'chain':hchain + [available_hops[new_node]], 
                                                'tfidf_mean': ((c+1)/(c+2))*tfidf_mean + (1/(c+1))*tfidf_scores[new_node]}
                    new_hybrid_chain_list.append(new_hchain)
            
            # picking top-beam_size hybrid chain
            # new_hybrid_chain_list += hybrid_chain_list 
            sorted_hybrid_chain_list = sorted(new_hybrid_chain_list, key=lambda chain:chain['tfidf_mean'])[::-1]
            hybrid_chain_list = sorted_hybrid_chain_list[:beam_size]
        
        hybrid_chain_text = []
        for hybrid_chain in hybrid_chain_list:
            hybrid_chain_text.append([row[node]['text'] for node in hybrid_chain['chain']])
        return hybrid_chain_list, hybrid_chain_text
    
    def hybrid_chains(self, data, beam_size=10, num_hops=[3,4], num_chains_per_hops=4):
        processed_data_dict = {'question': [], 'input': []}
        for qid, qdict in tqdm(enumerate(data), desc="Extracting chains"):
            row = qdict.pop('row')
            question_repr = self.tfid_model.transform([qdict['question']])
    
            row_parts = list(row.keys())
            all_text = [row[k]['text'] for k in row_parts]
    
            parts_repr = self.tfid_model.transform(all_text)
            sim_with_question = linear_kernel(question_repr, parts_repr)[0]
            
            for i, key in enumerate(row_parts):
                row[key]['tfidf_score'] = sim_with_question[i]

            answer_node = [key for key in row if qdict['answer_text'].lower() in row[key]['text'].lower()]

            #ignore if answer node is None
            if not answer_node: continue

            candidate_chains = []
            for hops in num_hops:
                chain_ids, chain_text = self.get_candidate_hybrid_chains(row, answer_node[0], hops, beam_size)
                for j in range(min(num_chains_per_hops, len(chain_ids))):
                    candidate_chains.append({'chain-text': chain_text[j], 'score': chain_ids[j]['tfidf_mean']})

            #ignore if no chain is extracted
            if not candidate_chains: continue
            qdict['hybrid_chain'] = candidate_chains
            
            t5_input_sequence, target = self.hybrid_chain_to_t5_sequence(qdict)
            processed_data_dict['question'].append(target)
            processed_data_dict['input'].append(t5_input_sequence)
        return processed_data_dict

    def link_sents_to_cells(self, qdict):
        # Split passages in to sentences. Sentences becomes nodes with 'text' and 'link'
        qdict = deepcopy(qdict)
        pkeys = [key for key in qdict if 'passage_' in key]
        for key in pkeys:
            passage_dict = qdict.pop(key)
            sents = self.nlp_model(passage_dict['text']).sentences
            sents = [sent.text for sent in sents]
        
            linked_cell = qdict[passage_dict['links']]
            linked_cell['links'].remove(key) # removing link to the passage from linked cell (will add sentence in for loop)
        
            ### Not adding links between sentences of same passage for now. 
            for i,s in enumerate(sents):
                sent_dict = {'text': s}
                sent_dict['links'] = [passage_dict['links']]

                sent_name = key + '_' + str(i)
                qdict[sent_name] = sent_dict
                linked_cell['links'].append(sent_name)
        return qdict
                        
    def preprocess_hybridqa_data(self, data):
        # identify passages linked to cells in the rows of table.
        # make a nice dict to store this "structured fused block"
        new_data = []
        all_text_list = []
        for sample in tqdm(data):
            question = sample['question']
            question_id = sample['question_id']
            answer_text = sample['answer_text']
            table_id = sample['table_id']

            #ignore if question or answer is None
            if not question.strip() or not answer_text.strip(): continue
            
            table = sample.pop('table')
            for tkey in ['url', 'title', 'section_title', 'section_text', 'uid', 'intro']: 
                sample[tkey] = table[tkey]

            header = table['header']
            num_cols = len(header)
            tabular_data = table['data']
            num_rows = len(tabular_data)//num_cols
            
            rows = np.split(np.array(tabular_data), num_rows)
            
            linked_cell_keys = range(num_cols)
            
            all_text_list.extend([question, sample['intro']])
    
            sample['rows'] = []
            for row_id, row in enumerate(rows):
                row_dict = {}
                passage_counter = 0
                all_row_text = []
                answer_row_ids = []
                for i, cell in enumerate(row):
                    value = cell['value']
                    col_name = header[i]
                    cell_text = f'The {col_name} is {value}.'
                    urls = cell['urls']
                    row_dict[f'cell_{i}'] = {}
                    row_dict[f'cell_{i}']['text'] = cell_text
                    all_row_text.append(cell_text)
                    all_text_list.append(cell_text)
                    row_dict[f'cell_{i}']['links'] = [f'cell_{j}' for j in linked_cell_keys if i != j]
    
                    #process only cells have linked passages
                    for psg_url in urls:
                        summary = psg_url['summary']
                        passage_url = psg_url['url']
                        all_row_text.append(summary)
                        all_text_list.append(summary)
                        row_dict[f'passage_{passage_counter}'] = {}
                        row_dict[f'passage_{passage_counter}']['text'] = summary
                        row_dict[f'passage_{passage_counter}']['links'] = f'cell_{i}'
                        row_dict[f'cell_{i}']['links'].append(f'passage_{passage_counter}')
                        passage_counter += 1
    
                #indentify answer positions in table and its linked passages
                all_row_text = ' '.join(all_row_text).lower()
                if answer_text.lower() in all_row_text:
                    answer_row_ids.append([row_id, all_row_text.count(answer_text.lower())])

                sample['rows'].append(row_dict)
                
                # if answer occurs just once in the row text
                if len(answer_row_ids) == 1 and answer_row_ids[0][1] == 1:
                    sample['row'] = sample['rows'][answer_row_ids[0][0]] # answer row
                    #split the passage and link sentences to cells individualy
                    sample['row'] = self.link_sents_to_cells(sample['row'])

            del sample['rows']
            if sample.get('row'):
                new_data.append(sample)
        return new_data, all_text_list

    def preprocess_data_for_qg(self, data_split='train', beam_size=10, num_hops=[3,4], num_chains_per_hops=4):
        data = load_dataset(self.dataset_name, split=data_split)
        processed_data_dict = {'question': [], 'input': []}
        data, all_text = self.preprocess_hybridqa_data(data)
        self.tfid_model = TfidfVectorizer().fit(all_text)
        data = self.hybrid_chains(data, beam_size=beam_size, num_hops=num_hops, num_chains_per_hops=num_chains_per_hops)
        return data
