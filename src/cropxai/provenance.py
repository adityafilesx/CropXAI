"""Immutable JSON records and additive SQLite migration for revision runs."""
from __future__ import annotations
from contextlib import closing
import hashlib, json, sqlite3, platform, importlib.metadata
from pathlib import Path
from datetime import datetime, timezone

def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(value): return hashlib.sha256(canonical(value).encode()).hexdigest()
def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()
def source_hash(root=Path('.')):
    # Only the modules that determine a run's numerical output. Editing analysis, figure, paper or
    # legacy code must not change run identity, otherwise one cohort would split into groups.
    return digest({str(p):file_hash(p) for p in sorted(Path(root,'src','cropxai').rglob('*.py')) if p.name!='cli.py' and '__pycache__' not in p.parts})
def environment():
    import torch
    return {'python':platform.python_version(),'platform':platform.platform(),'machine':platform.machine(),
            'processor':platform.processor(),'cuda':torch.cuda.is_available(),'mps':torch.backends.mps.is_available(),
            'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}
def write_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    text=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(text); tmp.replace(path)
def append_record(record, db_path='results/experiment_db.sqlite'):
    required={'schema_version','run_id','status','timestamp','config','config_hash','source_revision','dataset','model','seed','environment','artifacts','metrics'}
    if not required<=set(record): raise ValueError('Incomplete provenance record')
    if record['config_hash']!=digest(record['config']): raise ValueError('Configuration hash mismatch')
    if record['status'] not in ('completed','failed'): raise ValueError('Terminal status required')
    payload=canonical(record) # rejects NaN
    with closing(sqlite3.connect(db_path)) as conn, conn:
        conn.execute('CREATE TABLE IF NOT EXISTS revision_schema (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS revision_runs (run_id TEXT PRIMARY KEY, status TEXT NOT NULL, record_json TEXT NOT NULL, sha256 TEXT NOT NULL)')
        conn.execute('INSERT OR IGNORE INTO revision_schema VALUES (?,?)',(2,datetime.now(timezone.utc).isoformat()))
        conn.execute('INSERT INTO revision_runs VALUES (?,?,?,?)',(record['run_id'],record['status'],payload,hashlib.sha256(payload.encode()).hexdigest()))
