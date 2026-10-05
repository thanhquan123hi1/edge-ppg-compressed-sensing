"""File and numerical provenance with atomic writes for resumable runs."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

def file_sha256(path):
    h = hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):
            h.update(block)
    return h.hexdigest()

def contract_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def array_hash(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode())
    h.update(memoryview(a).cast('B'))
    return h.hexdigest()

def save_json(path,value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')
    os.replace(tmp,path)

def create_run_manifest(config,source_paths):
    from dataclasses import asdict
    return {'run_id':config.run_id,'config':asdict(config),'numerical_contract':config.numerical_contract(),
            'protocol_hash':contract_hash(config.numerical_contract()),
            'source_hashes':{str(p):file_sha256(p) for p in source_paths},
            'test_disclosure':'Existing PPG-DaLiA test benchmark reviewed before extension; no pristine holdout claim.'}
