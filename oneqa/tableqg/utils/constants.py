
from dataclasses import dataclass
# constants for SQL in WikiSQL

@dataclass
class SqlOperants:
	agg_ops = ['select', 'maximum', 'minimum', 'count', 'sum', 'average']
	cond_ops = ['=', '>', '<', 'OP']
	cond_ops_string = ['equal', 'greater', 'lesser', 'OP']

@dataclass
class T5SpecialTokens:
	sep = ' <extra_id_0> '
	cond = ' <extra_id_1> '
	ans = ' <extra_id_2> '
	header = ' <extra_id_3> '
	hsep = ' <extra_id_4> '