import sys, json, shutil, importlib.util
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.cropxai.provenance import environment, write_json
if __name__=='__main__':
    result=environment()
    result['tools']={k:shutil.which(k) for k in ['pdflatex','bibtex','pdftoppm','pdftotext']}
    result['optional_packages']={k:bool(importlib.util.find_spec(k)) for k in ['pytest','transformers','captum','pypdf']}
    write_json('docs/revision/evidence/environment.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='packages'},indent=2))
