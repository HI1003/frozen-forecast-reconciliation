"""Local reader-run storage: pinned inputs, complete candidates and frozen choices.

A run is an independent directory. It never loads the author's raw archive.
Interrupted candidate/test directories are deliberately not retried or overwritten.
"""
from __future__ import annotations
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path

from scripts.experiments.paper_protocol import ROOT, DATASETS, BASES, BUDGET_SELECTION, unit_id


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    """Exclusive creation; NaNs and silent replacement are forbidden."""
    raw=json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n'
    with Path(path).open('x',encoding='utf-8') as stream: stream.write(raw)


@contextmanager
def run_lock(run):
    from src.support.storage.locking import lock_file,unlock_file
    run=Path(run);run.mkdir(parents=True,exist_ok=True)
    with (run/'.lock').open('a+',encoding='utf-8') as stream:
        lock_file(stream,blocking=False)
        try: yield
        finally: unlock_file(stream)


def initialise(run, inputs):
    run=Path(run);inputs=Path(inputs).resolve()
    mapping=read(inputs)
    if set(mapping)!={d+'/'+b for d in DATASETS for b in BASES}:
        raise ValueError('inputs must map exactly four dataset/base names to generated .npz files')
    paths={k:str((inputs.parent/v).resolve()) for k,v in mapping.items()}
    files={str(inputs):sha(inputs)}
    for value in paths.values():
        p=Path(value)
        if p.suffix!='.npz': raise ValueError('expected .npz forecasts')
        for f in (p,p.with_suffix('.meta.json')): files[str(f)]=sha(f)
    # Bind actual public implementation/config/data bytes, independent of Git state.
    source={}
    source[BUDGET_SELECTION.relative_to(ROOT).as_posix()]=sha(BUDGET_SELECTION)
    for directory,pattern in [('src','*.py'),('scripts','*.py'),('configs','*'),('Data','*')]:
        for p in sorted((ROOT/directory).rglob(pattern)):
            if p.is_file() and '__pycache__' not in p.parts: source[p.relative_to(ROOT).as_posix()]=sha(p)
    value=dict(schema=1,inputs=paths,input_hashes=files,source_hashes=source)
    manifest=run/'run.json'
    if manifest.exists():
        if read(manifest)!=value: raise ValueError('run inputs or implementation changed; use a new run directory')
    else: write(manifest,value)
    return value


def verify_run(run):
    record=read(Path(run)/'run.json')
    for path,digest in record['input_hashes'].items():
        if sha(path)!=digest: raise ValueError('changed frozen input: '+path)
    for path,digest in record['source_hashes'].items():
        if sha(ROOT/path)!=digest: raise ValueError('changed implementation/config/data: '+path)
    return record


def require_pretest(run):
    if (Path(run)/'test.started.json').exists():
        raise ValueError('test access has started; fitting and selection are closed')


def candidate_dir(run, u):
    return Path(run)/'candidates'/unit_id(u)


def complete_candidate(folder, value):
    folder=Path(folder)
    write(folder/'validation.json',value)
    files={p.relative_to(folder).as_posix():sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    write(folder/'complete.json',dict(files=files))


def candidate_record(run,u):
    folder=candidate_dir(run,u)
    receipt=read(folder/'complete.json')
    for p,digest in receipt['files'].items():
        if sha(folder/p)!=digest: raise ValueError('changed candidate: '+p)
    row=read(folder/'validation.json')
    if row['unit']!=u: raise ValueError('candidate identity mismatch')
    return row


def freeze(run,selection):
    run=Path(run);require_pretest(run)
    files={}
    for u in selection['units']:
        candidate_record(run,u)
        p=candidate_dir(run,u)/'complete.json'
        files[p.relative_to(run).as_posix()]=sha(p)
    write(run/(selection['group']+'.selection.json'),dict(selection,files=files))


def selection(run,group):
    run=Path(run);r=read(run/(group+'.selection.json'))
    for p,digest in r['files'].items():
        if sha(run/p)!=digest: raise ValueError('selected checkpoint receipt changed')
    return r


def frozen_units(run, *, start=False, confirmed=False, preview=False):
    run=Path(run)
    if start and not confirmed: raise ValueError('pass --confirm-selection-frozen only after reviewing validation choices')
    records={g:selection(run,g) for g in ('core','penalty','budget')}
    units={unit_id(u):u for s in records.values() for u in s['units']}
    receipt={g:sha(run/(g+'.selection.json')) for g in records}
    marker=run/'test.started.json'
    if marker.exists():
        if read(marker)!=receipt: raise ValueError('selection changed after test access')
    elif start: write(marker,receipt)
    elif not preview: raise ValueError('evaluation has not started')
    return list(units.values())


def begin_test(run,u):
    """Claim before any test array read. A failed test is not automatically rerun."""
    folder=Path(run)/'test'/unit_id(u)
    folder.mkdir(parents=True,exist_ok=False)
    write(folder/'started.json',dict(unit=u,checkpoint_receipt_sha256=sha(candidate_dir(run,u)/'complete.json')))
    return folder


def load_pair(record,dataset,base,segments=('train','val')):
    from src.data.datasets import load_config,load_dataset
    from src.support.storage.forecasts import load_base_forecasts
    data=load_dataset(load_config(ROOT/'configs/data'/f'{dataset}.yaml'),ROOT)
    arrays=load_base_forecasts(record['inputs'][dataset+'/'+base],data,base,segments=segments)
    return data,arrays


def cpu_state_dict(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
