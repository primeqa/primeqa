import random
import numpy as np

from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

import stanza

from primeqa.qg.utils.constants import QGSpecialTokens


class PathSampler():
    """
    Samples hybrid chains from hybrid context of table and text. The sampled chains act as an input
    for HybridQG inference to generate relevant questions.
    
    Example chain: 
             <answer> Mactan-Cebu International Airport </answer>
             <chain> Mactan-Cebu International Airport located on Mactan Island , is the second busiest 
            airport in the Philippines . <sep> The Island is Cebu. <hsep> The Population is 3,979,155.</chain>

    The sampled chains have two parts, one a named entity or a cell text as a possible answer from the hybird 
    context, and the other a set of sentences from the context which contain some reference to the named entity. 
    We use NER from stanza library to extract named entities. Currently we ONLY support English!
    """
    def __init__(self, lang):
        if lang != 'en':
            raise NotImplementedError('This HybridQG for %s language is not supported.' % (lang.upper()))

        self.nlp_model = stanza.Pipeline(lang='en', processors='tokenize,pos,ner', verbose=False)
        self.tfidf = TfidfVectorizer()
    
    def tokenize_text(self, text):
        doc = self.nlp_model.process(text, processors='tokenize')
        sents = []
        for sent in doc.sentences:
            sent = [word.text for word in sent.tokens]
            sents.append(" ".join(sent))
        return sents

    def aggregate_data(self, table_data, passages):
        header = [h for h,_ in table_data['header']]
        rows = []
        all_text = []
        for row_list in table_data['data']:
            row = []
            for hid, (cell, links) in enumerate(row_list):
                col_name = header[hid]
                #NOTE ignore if column header/cell is null or free text in cell
                if not col_name.strip() or not cell.strip() or len(cell.split()) > 6: continue
                text = '\n'.join([passages.get(link, str()) for link in links])
                text_sents = self.tokenize_text(text)
                all_text.extend(text_sents)
                cell_text = "The %s is %s . " % (col_name, cell)
                all_text.append(cell_text)
                row.append((col_name, cell, text_sents))
            rows.append(row)
            
        return {'table': rows, 'text': all_text}

    def create_chains(self, row, answer_length=4):
        def create_chain(answer, answer_context, hop2, hop3):
            return {'answer': answer,
                     'answer_context': answer_context, 
                     'hop2': hop2, 
                     'hop3': hop3, 
                    }
        
        chains = []
        seen_chains = set([])
        for idx, (cell_p1, similar_sent_p1, entities_p1) in enumerate(row):
            cell_id1, cell_p1 = cell_p1[0], cell_p1[1:]
            cell_text_p1 = 'The %s is %s .' % cell_p1
            for (cell_p2, similar_sent_p2, entities_p2) in row[idx+1:]:
                if similar_sent_p1 == similar_sent_p2: continue
                cell_id2, cell_p2 = cell_p2[0], cell_p2[1:]
                cell_text_p2 = 'The %s is %s .' % cell_p2
                if cell_id1 == cell_id2: continue

                if similar_sent_p1:
                    answer = cell_p2[-1]
                    if len(answer.split()) <= answer_length and answer not in seen_chains:
                        chain = create_chain(answer, cell_text_p2, cell_text_p1, similar_sent_p1)
                        chains.append(chain)
                        seen_chains.add(answer)

                if similar_sent_p2:
                    answer = cell_p1[-1]
                    if len(answer.split()) <= answer_length and answer not in seen_chains:
                        chain = create_chain(answer, cell_text_p1, cell_text_p2, similar_sent_p2)
                        chains.append(chain)
                        seen_chains.add(answer)
        
                for answer in set(entities_p1):
                    if len(answer.split()) <= answer_length and answer not in seen_chains:
                        chain = create_chain(answer, similar_sent_p1, cell_text_p1, cell_text_p2)
                        chains.append(chain)
                        seen_chains.add(answer)

                for answer in set(entities_p2):
                    if len(answer.split()) <= answer_length and answer not in seen_chains:
                        chain = create_chain(answer, similar_sent_p2, cell_text_p1, cell_text_p2)
                        chains.append(chain)
                        seen_chains.add(answer)
        return chains
            
    def sample_paths(self, tabular_data, threshold=0.25, num_paths=5):
        paths = []
        picked_rows = []
        range_of_rows = range(len(tabular_data))
        while len(paths) < num_paths and len(picked_rows) < len(tabular_data):
            row_id = random.choice(range_of_rows)
            if row_id in picked_rows: continue
            picked_rows.append(row_id)
            row = tabular_data[row_id]
            processed_row = []
            for cell_id, (col_name, cell, passage_sents) in enumerate(row):
                if (col_name.strip() is None) or (cell.strip() is None) or (cell in ['none', 'None']): continue
                named_cell_text = 'The %s is %s .' % (col_name, cell)
                if not passage_sents:
                    processed_row.append(((cell_id, col_name, cell), str(), []))
                else:
                    cell_repr = self.tfidf_model.transform([named_cell_text])
                    passage_repr = self.tfidf_model.transform(passage_sents)
                    scores = linear_kernel(passage_repr, cell_repr)

                    filtered_ids = scores > threshold
                    similar_sentences = np.matrix(passage_sents).reshape(filtered_ids.shape[0], -1)[filtered_ids].reshape(-1).tolist()[0]
                    if not similar_sentences:
                        processed_row.append(((cell_id, col_name, cell), str(), []))
                    else:
                        for sent in similar_sentences:
                            nlp_output = self.nlp_model(sent).sentences[0]
                            named_entities = set([ent.text for ent in nlp_output.ents if ent.type not in ['ORDINAL', 'CARDINAL', 'NORP']])
                            processed_row.append(((cell_id, col_name, cell), sent, named_entities))    

            chains = self.create_chains(processed_row)
            if chains:
                choose_k = np.random.choice(range(1, min(len(chains) + 1, num_paths)), replace=False)
                picked_chains = np.random.choice(chains, choose_k, replace=False)
                paths.extend(picked_chains)

        if len(paths) > num_paths:
            return np.random.choice(paths, num_paths, replace=False)
        return paths
            
    def create_qg_input(self, 
                        data_list, 
                        num_questions_per_instance = 5, 
                        id_list=[]):
        """
        creates input for qg inference. Samples named entities or cell text
        as possible answers from a hybird context i.e., a table and its linked text passages.

        Args:
            data_list (list): List of tuples, each tuple contains two elements; a table and its passages.
            num_questions_per_instance (int): How many chains to sample from the hybrid input.
        Returns:
            input_str_list (list): List of hybrid chains sampled.
            ans_list (list): List of answers sampled.
        """
        ans_list = []
        all_paths = []
        input_str_list = []
        id_question_list = []
        answer_separator = " %s " % (QGSpecialTokens.ans)
        hop_separator = " %s " % (QGSpecialTokens.hsep)
        meta_separator = " %s " % (QGSpecialTokens.header)

        for i, (table, passages) in enumerate(data_list):
            if not table:
                raise AssertionError("Table is empty.")

            if not passages:
                raise AssertionError("Passages is empty.")

            uid = table['uid']
            title = table['title']
            section_title = table['section_title']

            aggregated_data = self.aggregate_data(table, passages)
                 
            self.tfidf_model = self.tfidf.fit(aggregated_data['text'])

            agg_tabular_data = aggregated_data['table']
            paths = self.sample_paths(agg_tabular_data, num_paths=num_questions_per_instance)

            for path in paths:
                ans = path['answer']
                chain = [path['answer_context'],
                            path['hop2'],
                            path['hop3'] 
                        ]

                meta = [title, section_title]
                chain_with_separators = hop_separator.join(chain)
                meta_with_separators = meta_separator.join([' ']+meta)
                text = "%s %s %s %s" % (ans, answer_separator, chain_with_separators.strip(), meta_with_separators.strip())	
                input_str_list.append(text.lower())
                ans_list.append(ans.lower())
                if len(id_list) > i and id_list[i] != None :
                    id_question_list.append(id_list[i])

        return input_str_list, ans_list, id_question_list
