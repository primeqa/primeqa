import json
from tqdm import tqdm

def old_map():
    with open("output.json") as inp:
        data = json.load(inp)
        for d in tqdm(data):
            url = d['document_url']
            id = d['document_id']
            loio = url.split('/')[-1].replace(".html",'')
            print (f"{id}\t{loio}")

def create_docid_map(fname, ofname):
    docid_map = {}
    with open(fname) as inp:
        for line in inp:
            d = json.loads(line)
            url = d['document_url']
            id = d['document_id']
            version = d['versionId']
            try:
                float(version)
            except ValueError:
                version = ".".join(version.split(" "))

            loio = url.split('/')[-1].split('?')[0].replace(".html",'')      
            if loio in docid_map:
                print (loio, docid_map[loio], version)
            if loio not in docid_map or float(docid_map[loio]['version']) < float(version):
                if loio in docid_map:
                    print ("replacing")
                docid_map[loio] = {"docid":id, "version":version}
   
    final_map = {}
    
    
    with open(ofname, 'w') as op:
        for k, v in docid_map.items():
            op.write(v['docid']+"\t"+k+"\n") 
    
            
create_docid_map("/u/raduf/sandbox2/5lang_docs/sap_es_docs.s4h_sfsf.jsonl", "/u/raduf/sandbox2/5lang_docs/es_docid.tsv")