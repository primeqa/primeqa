import pytest
from datasets import load_dataset
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.table_qg.sql_processor import SqlProcessor

test_data = [({'header': ['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team'], 'page_title': 'Toronto Raptors all-time roster', 'page_id': '', 'types': ['text', 'text', 'text', 'text', 'text', 'text'], 'id': '1-10015132-11', 'section_title': 'L', 'caption': 'L', 'rows': [['Antonio Lang', '21', 'United States', 'Guard-Forward', '1999-2000', 'Duke'], ['Voshon Lenard', '2', 'United States', 'Guard', '2002-03', 'Minnesota'], ['Martin Lewis', '32, 44', 'United States', 'Guard-Forward', '1996-97', 'Butler CC (KS)'], ['Brad Lohaus', '33', 'United States', 'Forward-Center', '1996', 'Iowa'], ['Art Long', '42', 'United States', 'Forward-Center', '2002-03', 'Cincinnati'], ['John Long', '25', 'United States', 'Guard', '1996-97', 'Detroit'],['Kyle Lowry', '3', 'United States', 'Guard', '2012-Present', 'Villanova']]},
            {'human_readable': 'SELECT Position FROM table WHERE School/Club Team = Butler CC (KS)', 'sel': 3, 'agg': 0, 'conds': {'column_index': [5], 'operator_index': [0], 'condition': ['Butler CC (KS)']}},
            ['guard-forward'])]

@pytest.mark.parametrize("table,sql,answer",test_data)
def test_sql_execute(table,sql,answer):
    assert SqlProcessor._execute_sql(sql,table) == answer

string_test_data = [({'header': ['Player', 'No.', 'Nationality', 'Position', 'Years in Toronto', 'School/Club Team'], 'page_title': 'Toronto Raptors all-time roster', 'page_id': '', 'types': ['text', 'text', 'text', 'text', 'text', 'text'], 'id': '1-10015132-11', 'section_title': 'L', 'caption': 'L', 'rows': [['Antonio Lang', '21', 'United States', 'Guard-Forward', '1999-2000', 'Duke'], ['Voshon Lenard', '2', 'United States', 'Guard', '2002-03', 'Minnesota'], ['Martin Lewis', '32, 44', 'United States', 'Guard-Forward', '1996-97', 'Butler CC (KS)'], ['Brad Lohaus', '33', 'United States', 'Forward-Center', '1996', 'Iowa'], ['Art Long', '42', 'United States', 'Forward-Center', '2002-03', 'Cincinnati'], ['John Long', '25', 'United States', 'Guard', '1996-97', 'Detroit'],['Kyle Lowry', '3', 'United States', 'Guard', '2012-Present', 'Villanova']]},
            {'human_readable': 'SELECT Position FROM table WHERE School/Club Team = Butler CC (KS)', 'sel': 3, 'agg': 0, 'conds': {'column_index': [5], 'operator_index': [0], 'condition': ['Butler CC (KS)']}},
            ['guard-forward'],"select <<sep>> Position <<sep>> School Club Team <<cond>> equal <<cond>> Butler CC (KS) <<answer>> ['guard-forward'] <<header>> Player <<hsep>> No. <<hsep>> Nationality <<hsep>> Position <<hsep>> Years in Toronto <<hsep>> School/Club Team")]

@pytest.mark.parametrize("table,sql,answer,sql_string",string_test_data)            
def test_create_sql_string(sql,table,answer,sql_string):
    assert SqlProcessor._create_sql_string(sql,table,answer) == sql_string

@pytest.mark.parametrize("model_name",["t5-small"])
def test_preprocess_data_for_qg(model_name):
    # model might change tokenizer
    tokenizer = QGModel(model_name, modality='table').tokenizer
    input_max_len = 1024
    target_max_len = 1024
    
    processor = SqlProcessor(tokenizer, input_max_len, target_max_len)
    dataset = load_dataset('wikisql', name='secondary_task', split='validation')
    processed_dataset = processor(dataset)
    assert processed_dataset != None
    assert len(processed_dataset['question']) == len(processed_dataset['input'])
    