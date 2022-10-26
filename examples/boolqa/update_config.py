import json
import sys

# load json
with open('primeqa/boolqa/tydi_boolqa_config.json', 'r') as f:
    data = json.load(f)
# update json
data['sn']['model_name_or_path'] = sys.argv[1]
# dump json
with open('primeqa/boolqa/tydi_boolqa_config.json', 'w') as f:
    json.dump(data, f, indent=4)