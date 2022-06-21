
from dataclasses import dataclass
# constants for SQL in WikiSQL

@dataclass
class SqlOperants:
	agg_ops = ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
	cond_ops = ['=', '>', '<', 'OP']
	cond_ops_string = ['equal', 'greater', 'lesser', 'OP']

@dataclass
class QGSpecialTokens:
	sep = '<<sep>>'
	cond = '<<cond>>'
	ans = '<<answer>>'
	header = '<<header>>'
	hsep = '<<hsep>>'