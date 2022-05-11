import pytest
from oneqa.tableqg.models.sql_sampler import SimpleSqlSampler

class TableQG_Test:

    def __init__(self):
        self.table = {"header": ["Player", "No.", "Nationality", "Position", "Years in Toronto", "School Team"],
                "rows": [
            ["Antonio Lang", 21, "United States", "Guard-Forward", "1999-2000", "Duke"],
            ["Voshon Lenard", 2, "United States", "Guard", "2002-03", "Minnesota"],
            ["Martin Lewis", 32, "United States", "Guard-Forward", "1996-97", "Butler CC (KS)"],
            ["Brad Lohaus", 33, "United States", "Forward-Center", "1996", "Iowa"],
            ["Art Long", 42, "United States", "Forward-Center", "2002-03", "Cincinnati"]
            ],
            "types":["text","real","text","text","text","text"]
            }
    

    def test_add_column_types(self):
        sql_sampler = SimpleSqlSampler()
        assert sql_sampler.add_column_types(self.table)['types'] == ['text', 'real', 'text', 'text', 'text', 'text']

