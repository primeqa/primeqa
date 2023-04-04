from nltk.tokenize import word_tokenize, sent_tokenize
import urllib.parse
import sys
import json
import gzip
from dateutil.parser import parse
import re
import warnings
warnings.filterwarnings("ignore")


def tokenize(string, remove_dot=False):
    def func(string):
        return " ".join(word_tokenize(string))
    string = string.replace('%-', '-')
    if remove_dot:
        string = string.rstrip('.')

    string = func(string)
    string = string.replace(' %', '%')
    return string

def url2dockey(string):
    string = urllib.parse.unquote(string)
    return string

def filter_firstKsents(string, k):
    string = sent_tokenize(string)
    string = string[:k]
    return " ".join(string)

def compressGZip(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)

    json_str = json.dumps(data) + "\n"               # 2. string (i.e. JSON)
    json_bytes = json_str.encode('utf-8')            # 3. bytes (i.e. UTF-8)

    with gzip.GzipFile(file_name + '.gz', 'w') as fout:   # 4. gzip
        fout.write(json_bytes)

def readGZip(file_name):
    if file_name.endswith('gz'):
        with gzip.GzipFile(file_name, 'r') as fin:    # 4. gzip
            json_bytes = fin.read()                      # 3. bytes (i.e. UTF-8)

        json_str = json_bytes.decode('utf-8')            # 2. string (i.e. JSON)
        data = json.loads(json_str)                      # 1. data
        return data
    else:
        with open(file_name, 'r') as fin:
            data = json.load(fin)
        return data


class CellHelper(object):
  """Cell Helper to detect the cell type."""

  @staticmethod
  def is_unit(string):
    """Is the input a unit."""
    return re.search(r'\b(kg|m|cm|lb|hz|million)\b', string.lower())

  @staticmethod
  def is_score(string):
    """Is the input a score between two things."""
    if re.search(r'[0-9]+ - [0-9]+', string):
      return True
    elif re.search(r'[0-9]+-[0-9]+', string):
      return True
    else:
      return False

  @staticmethod
  def is_date(string, fuzzy=False):
    """Is the input a date."""
    try:
      parse(string, fuzzy=fuzzy)
      return True
    except Exception:  # pylint: disable=broad-except
      return False

  @staticmethod
  def is_bool(string):
    if string.lower() in ['yes', 'no']:
      return True
    else:
      return False

  @staticmethod
  def is_float(string):
    if '.' in string:
      try:
        float(string)
        return True
      except Exception:
        return False
    else:
      return False

  @staticmethod
  def is_normal_word(string):
    if ' ' not in string:
      return string.islower()
    else:
      return False


def whitelist(string):
  """Is the input a whitelist string."""
  string = string.strip()
  if len(string) < 2:
    return False
  elif string.isdigit():
    if len(string) == 4:
      return True
    else:
      return False
  elif string.replace(',', '').isdigit():
    return False
  elif CellHelper.is_float(string):
    return False
  elif '#' in string or '%' in string or '+' in string or '$' in string: 
    return False
  elif CellHelper.is_normal_word(string):
    return False
  elif CellHelper.is_bool(string):
    return False
  elif CellHelper.is_score(string):
    return False
  elif CellHelper.is_unit(string):
    return False
  elif CellHelper.is_date(string):
    return False
  return True


def is_year(string):
  if len(string) == 4 and string.isdigit():
    return True
  else:
    return False